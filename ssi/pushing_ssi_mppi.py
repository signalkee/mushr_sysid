"""
SSI-MPPI: Simultaneous System Identification + Model Predictive Path Integral

Combines:
- MPPI: Sampling-based MPC (from path_tracking_controller.py)
- SSI: Online learning of unknown mass/friction

The dynamics model used in MPPI rollouts is augmented with SSI residuals.
Uses same 11D state and acceleration control as SSI MPC for consistency.
"""

import numpy as np
import torch
from pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal
from .ackermann_model import AckermannCar


class PushingSSIMPPI:
    """
    SSI-MPPI controller for pushing with online learning.
    
    Uses same 11D state and acceleration control as SSI MPC:
    State: [x_car, y_car, theta_car, v_car, omega_car, 
            x_block, y_block, theta_block, vx_block, vy_block, omega_block]  (11D)
    Control: [steering_angle, acceleration]  (2D)
    
    SSI learns residuals in block accelerations to compensate for unknown mass and friction.
    """
    
    def __init__(self,
                 device='cpu',
                 n_rf=20,
                 lr=0.2,
                 kernel_std=0.3,
                 nominal_mass=0.3,
                 nominal_friction=0.2,
                 true_mass=0.8,
                 true_friction=0.5):
        """
        Initialize SSI-MPPI controller.
        
        Args:
            device: torch device
            n_rf: Number of random features
            lr: Learning rate for SSI
            kernel_std: Gaussian kernel bandwidth
            nominal_mass: Wrong mass (used in dynamics)
            nominal_friction: Wrong friction (used in dynamics)
            true_mass: Actual mass (for documentation)
            true_friction: Actual friction (for documentation)
        """
        self.device = torch.device(device)
        self.trajectory = None
        self.index = 0
        
        # Car model (same as SSI MPC)
        self.car = AckermannCar()
        self.car.block_mass_nominal = nominal_mass
        self.car.block_friction_nominal = nominal_friction
        
        # SSI parameters
        self.n_rf = n_rf
        self.lr = lr
        self.kernel_std = kernel_std
        
        # Mass/friction (for documentation)
        self.nominal_mass = nominal_mass
        self.nominal_friction = nominal_friction
        self.true_mass = true_mass
        self.true_friction = true_friction
        
        print(f"   SSI-MPPI Nominal: mass={nominal_mass} kg, friction={nominal_friction}")
        print(f"   SSI-MPPI True:    mass={true_mass} kg, friction={true_friction}")
        
        # State and control dimensions (same as SSI MPC)
        self.state_dim = 11  # [car(5), block(6)]
        self.u_dim = 2  # [steering, acceleration]
        
        # Random features (same as SSI MPC)
        self.input_mask = list(range(11))  # Use all states as features
        self.target_mask = [8, 9, 10]  # Learn residuals for block velocities [vx_block, vy_block, omega_block]
        
        # Random features
        n_inputs = len(self.input_mask)
        self.omega = np.random.normal(0.0, kernel_std, (n_rf, n_inputs))
        self.b = np.random.uniform(0.0, 2.0 * np.pi, (n_rf, 1))
        
        # Mapping matrices (same as SSI MPC)
        self.Bh = np.eye(self.state_dim)[self.target_mask].T  # Map learned residuals to state space
        self.Bz = np.eye(self.state_dim + self.u_dim)[self.input_mask]  # Map to feature space
        
        # Learning parameters
        self.alpha = np.zeros((len(self.target_mask), n_rf))
        self.x_last = None
        self.u_last = None
        
        # Integration timestep (same as SSI MPC)
        self.dt = 0.05  # Typically horizon / num_steps = 0.5 / 10 = 0.05
        
        # MPPI setup
        self.reset_mppi()
        
    def reset_mppi(self):
        """Initialize MPPI controller."""
        TIMESTEPS = 50
        N_SAMPLES = 300
        # Control bounds: [steering, acceleration]
        ACTION_LOW = [-0.34, -0.5]  # min_steering, min_acceleration
        ACTION_HIGH = [0.34, 0.5]   # max_steering, max_acceleration
        
        nx = 11  # state dimension (11D)
        # Noise for [steering, acceleration]
        noise_sigma = torch.tensor([[0.5, 0.0],
                                    [0.0, 0.5]], device=self.device, dtype=torch.float32)
        lambda_ = 0.1
        noise_mu = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float32)
        
        self.ctrl = MPPI(
            self.push_dynamics_ssi,  # Use SSI-augmented dynamics
            self.running_cost, 
            nx=nx, 
            noise_sigma=noise_sigma,
            num_samples=N_SAMPLES, 
            horizon=TIMESTEPS,
            lambda_=lambda_, 
            device=self.device, 
            terminal_state_cost=self.terminal_state_cost, 
            noise_mu=noise_mu,
            u_min=torch.tensor(ACTION_LOW, dtype=torch.float32, device=self.device),
            u_max=torch.tensor(ACTION_HIGH, dtype=torch.float32, device=self.device),
            noise_abs_cost=False,
            sample_null_action=False
        )
    
    def compute_random_features(self, states, actions):
        """Compute random Fourier features (numpy version for SSI update)."""
        # Combine states and actions
        Z = np.hstack([states, actions]).reshape(-1, 1)  # (13 x 1)
        # Select features based on input mask
        Z_masked = self.Bz @ Z  # (len(input_mask) x 1)
        # Compute features
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ Z_masked + self.b)
        return rf
    
    def compute_random_features_torch(self, states, actions):
        """Compute random Fourier features (torch version for MPPI rollouts)."""
        # states: (N_samples, 11) tensor
        # actions: (N_samples, 2) tensor
        # Combine states and actions
        Z = torch.cat([states, actions], dim=1)  # (N_samples, 13)
        
        # Convert omega, b, and Bz to torch if needed
        if not isinstance(self.omega, torch.Tensor):
            omega_torch = torch.tensor(self.omega, dtype=torch.float32, device=states.device)
            b_torch = torch.tensor(self.b, dtype=torch.float32, device=states.device)
            Bz_torch = torch.tensor(self.Bz, dtype=torch.float32, device=states.device)
        else:
            omega_torch = self.omega.to(states.device)
            b_torch = self.b.to(states.device)
            Bz_torch = self.Bz.to(states.device)
        
        # Select features based on input mask: Bz @ Z^T
        # Z: (N_samples, 13), Bz: (len(input_mask), 13)
        # We need: Bz @ Z^T -> (len(input_mask), N_samples)
        Z_T = Z.T  # (13, N_samples)
        Z_masked = torch.mm(Bz_torch, Z_T)  # (len(input_mask), N_samples)
        
        # Compute features: (1/sqrt(n_rf)) * cos(omega @ Z_masked + b)
        # omega: (n_rf, len(input_mask)), Z_masked: (len(input_mask), N_samples)
        omega_Z = torch.mm(omega_torch, Z_masked)  # (n_rf, N_samples)
        omega_Z_b = omega_Z + b_torch  # (n_rf, N_samples)
        rf = (1.0 / np.sqrt(self.n_rf)) * torch.cos(omega_Z_b)  # (n_rf, N_samples)
        
        return rf  # (n_rf, N_samples)
    
    def ssi_update(self, dt, x_now):
        """
        SSI update step: gradient descent on prediction error.
        
        Same logic as SSI MPC update_step.
        
        Args:
            dt: Time since last update
            x_now: Current state (11D)
        """
        if self.x_last is None:
            self.x_last = x_now.copy()
            self.u_last = np.zeros(self.u_dim)
            return
        
        if dt == 0 or dt < 1e-6:
            dt = self.dt
        
        # Get previous data
        alpha_in = self.alpha
        x_in = self.x_last
        u_in = self.u_last
        
        # --- SSI Update: Gradient descent on prediction error ---
        
        # 1. Compute random features at previous time step
        Z = np.hstack((x_in, u_in)).reshape(-1, 1)  # (13 x 1)
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ (self.Bz @ Z) + self.b)  # (n_rf x 1)
        
        # 2. Predict next state using current alpha
        x_dot_pred = self.predict_next_state_numpy(x_in, u_in, alpha_in, dt)
        x_pred = x_in + dt * x_dot_pred
        
        # 3. Compute prediction error on target channels (block velocities)
        error_pred = self.Bh.T @ (x_pred.reshape(-1, 1) - x_now.reshape(-1, 1))  # (3 x 1)
        
        # 4. Gradient descent update
        alpha_out = alpha_in - 2.0 * self.lr * (error_pred @ rf.T)
        
        # 5. Store for next iteration
        self.alpha = np.copy(alpha_out)
        self.x_last = np.copy(x_now)
    
    def predict_next_state_numpy(self, states, actions, alpha, dt):
        """
        Predict next state using nominal dynamics + SSI residuals (numpy).
        
        Same dynamics as SSI MPC _augmented_dynamics.
        
        Args:
            states: 11D state vector
            actions: [steering, acceleration]
            alpha: SSI learning parameters
            dt: timestep
            
        Returns:
            x_dot: State derivatives (11D)
        """
        L = self.car.wheelbase
        offset = self.car.offset_to_front
        
        # Extract states
        x_car = states[0]
        y_car = states[1]
        theta_car = states[2]
        v_car = states[3]
        omega_car = states[4]
        x_block = states[5]
        y_block = states[6]
        theta_block = states[7]
        vx_block = states[8]
        vy_block = states[9]
        omega_block = states[10]
        
        steering = actions[0]
        accel = actions[1]
        
        # --- Car dynamics (Ackermann bicycle model) ---
        x_car_dot = v_car * np.cos(theta_car)
        y_car_dot = v_car * np.sin(theta_car)
        theta_car_dot = (v_car / L) * np.tan(steering)
        v_car_dot = accel
        omega_car_dot = (accel / L) * np.tan(steering)
        
        # --- Block dynamics (quasi-static nominal + SSI residuals) ---
        # Contact detection: block dynamics only apply when car is in contact with block
        bumper_x = x_car + offset * np.cos(theta_car)
        bumper_y = y_car + offset * np.sin(theta_car)
        
        # Contact parameters
        contact_threshold = 0.15  # meters - maximum distance for contact (block size ~0.1m + margin)
        min_push_velocity = 0.1  # m/s - minimum car velocity to push
        friction_coeff = 0.3  # Friction coefficient for block when not in contact
        
        # Distance from block to bumper
        dx_contact = x_block - bumper_x
        dy_contact = y_block - bumper_y
        contact_distance = np.sqrt(dx_contact**2 + dy_contact**2 + 1e-6)  # Add epsilon for numerical stability
        
        # Check if in contact: distance < threshold AND car is moving forward
        is_in_contact = 1.0 if contact_distance < contact_threshold else 0.0
        is_pushing = 1.0 if v_car >= min_push_velocity else 0.0
        contact_active = is_in_contact * is_pushing  # Both conditions must be true
        
        # Position derivatives (integrate from velocities - actual states)
        x_block_dot = vx_block
        y_block_dot = vy_block
        theta_block_dot = omega_block
        
        # Nominal velocity (quasi-static: block follows bumper) - only when in contact
        vx_block_nominal_dot = (v_car * np.cos(theta_car) - \
                               offset * theta_car_dot * np.sin(theta_car)) * contact_active
        vy_block_nominal_dot = (v_car * np.sin(theta_car) + \
                               offset * theta_car_dot * np.cos(theta_car)) * contact_active
        omega_block_nominal_dot = theta_car_dot * contact_active
        
        # Nominal accelerations (differentiating nominal velocities) - only when in contact
        vx_block_nominal_ddot = (accel * np.cos(theta_car) - \
                                v_car * theta_car_dot * np.sin(theta_car) - \
                                offset * (theta_car_dot**2 * np.cos(theta_car) + \
                                         omega_car_dot * np.sin(theta_car))) * contact_active
        
        vy_block_nominal_ddot = (accel * np.sin(theta_car) + \
                                v_car * theta_car_dot * np.cos(theta_car) - \
                                offset * (theta_car_dot**2 * np.sin(theta_car) - \
                                         omega_car_dot * np.cos(theta_car))) * contact_active
        
        omega_block_nominal_ddot = omega_car_dot * contact_active
        
        # Compute random features
        Z = np.hstack([states, actions]).reshape(-1, 1)
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ (self.Bz @ Z) + self.b)
        
        # SSI augmentation: learned residuals
        residuals = alpha @ rf  # (len(target_mask) x 1) = (3 x 1)
        
        # Apply residuals to block accelerations (only when in contact)
        # When not in contact, apply friction to decay block velocities
        vx_block_dot = (vx_block_nominal_ddot + residuals[0, 0]) * contact_active - \
                       friction_coeff * vx_block * (1.0 - contact_active)
        vy_block_dot = (vy_block_nominal_ddot + residuals[1, 0]) * contact_active - \
                       friction_coeff * vy_block * (1.0 - contact_active)
        omega_block_dot = (omega_block_nominal_ddot + residuals[2, 0]) * contact_active - \
                          friction_coeff * omega_block * (1.0 - contact_active)
        
        # Full state derivative
        x_dot = np.array([x_car_dot, y_car_dot, theta_car_dot, v_car_dot, omega_car_dot,
                          x_block_dot, y_block_dot, theta_block_dot,
                          vx_block_dot, vy_block_dot, omega_block_dot])
        
        return x_dot
    
    def push_dynamics_ssi(self, states, actions):
        """
        Push dynamics with SSI augmentation for MPPI rollouts.
        
        Exact same dynamics as SSI MPC _augmented_dynamics, implemented in PyTorch.
        
        Args:
            states: (N_samples, 11) tensor
            actions: (N_samples, 2) tensor [steering, acceleration]
            
        Returns:
            next_states: (N_samples, 11) tensor
        """
        dt = self.dt
        L = self.car.wheelbase
        offset = self.car.offset_to_front
        
        # Extract states
        x_car = states[:, 0]
        y_car = states[:, 1]
        theta_car = states[:, 2]
        v_car = states[:, 3]
        omega_car = states[:, 4]
        x_block = states[:, 5]
        y_block = states[:, 6]
        theta_block = states[:, 7]
        vx_block = states[:, 8]
        vy_block = states[:, 9]
        omega_block = states[:, 10]
        
        steering = actions[:, 0]
        accel = actions[:, 1]
        
        # --- Car dynamics (Ackermann bicycle model) ---
        x_car_dot = v_car * torch.cos(theta_car)
        y_car_dot = v_car * torch.sin(theta_car)
        theta_car_dot = (v_car / L) * torch.tan(steering)
        v_car_dot = accel
        omega_car_dot = (accel / L) * torch.tan(steering)
        
        # --- Block dynamics (quasi-static nominal + SSI residuals) ---
        # Contact detection: block dynamics only apply when car is in contact with block
        bumper_x = x_car + offset * torch.cos(theta_car)
        bumper_y = y_car + offset * torch.sin(theta_car)
        
        # Contact parameters
        contact_threshold = 0.15  # meters - maximum distance for contact (block size ~0.1m + margin)
        min_push_velocity = 0.1  # m/s - minimum car velocity to push
        friction_coeff = 0.3  # Friction coefficient for block when not in contact
        
        # Distance from block to bumper
        dx_contact = x_block - bumper_x
        dy_contact = y_block - bumper_y
        contact_distance = torch.sqrt(dx_contact**2 + dy_contact**2 + 1e-6)  # Add epsilon for numerical stability
        
        # Check if in contact: distance < threshold AND car is moving forward
        is_in_contact = (contact_distance < contact_threshold).float()
        is_pushing = (v_car >= min_push_velocity).float()
        contact_active = is_in_contact * is_pushing  # Both conditions must be true (N_samples,)
        
        # Position derivatives (integrate from velocities - actual states)
        x_block_dot = vx_block
        y_block_dot = vy_block
        theta_block_dot = omega_block
        
        # Nominal velocity (quasi-static: block follows bumper) - only when in contact
        vx_block_nominal_dot = (v_car * torch.cos(theta_car) - \
                               offset * theta_car_dot * torch.sin(theta_car)) * contact_active
        vy_block_nominal_dot = (v_car * torch.sin(theta_car) + \
                               offset * theta_car_dot * torch.cos(theta_car)) * contact_active
        omega_block_nominal_dot = theta_car_dot * contact_active
        
        # Nominal accelerations (differentiating nominal velocities) - only when in contact
        vx_block_nominal_ddot = (accel * torch.cos(theta_car) - \
                                v_car * theta_car_dot * torch.sin(theta_car) - \
                                offset * (theta_car_dot**2 * torch.cos(theta_car) + \
                                         omega_car_dot * torch.sin(theta_car))) * contact_active
        
        vy_block_nominal_ddot = (accel * torch.sin(theta_car) + \
                                v_car * theta_car_dot * torch.cos(theta_car) - \
                                offset * (theta_car_dot**2 * torch.sin(theta_car) - \
                                         omega_car_dot * torch.cos(theta_car))) * contact_active
        
        omega_block_nominal_ddot = omega_car_dot * contact_active
        
        # Compute random features for all samples in batch
        alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=states.device)
        rf = self.compute_random_features_torch(states, actions)  # (n_rf, N_samples)
        
        # Compute residuals: alpha @ rf -> (len(target_mask), N_samples)
        # alpha_torch: (len(target_mask), n_rf), rf: (n_rf, N_samples)
        residuals = torch.mm(alpha_torch, rf)  # (len(target_mask), N_samples)
        
        # Apply residuals to block accelerations (only when in contact)
        # When not in contact, apply friction to decay block velocities
        vx_block_dot = (vx_block_nominal_ddot + residuals[0, :]) * contact_active - \
                       friction_coeff * vx_block * (1.0 - contact_active)
        vy_block_dot = (vy_block_nominal_ddot + residuals[1, :]) * contact_active - \
                       friction_coeff * vy_block * (1.0 - contact_active)
        omega_block_dot = (omega_block_nominal_ddot + residuals[2, :]) * contact_active - \
                          friction_coeff * omega_block * (1.0 - contact_active)
        
        # Integrate state derivatives
        x_car_next = x_car + x_car_dot * dt
        y_car_next = y_car + y_car_dot * dt
        theta_car_next = theta_car + theta_car_dot * dt
        v_car_next = v_car + v_car_dot * dt
        omega_car_next = omega_car + omega_car_dot * dt
        
        x_block_next = x_block + x_block_dot * dt
        y_block_next = y_block + y_block_dot * dt
        theta_block_next = theta_block + theta_block_dot * dt
        vx_block_next = vx_block + vx_block_dot * dt
        vy_block_next = vy_block + vy_block_dot * dt
        omega_block_next = omega_block + omega_block_dot * dt
        
        next_states = torch.stack([
            x_car_next, y_car_next, theta_car_next, v_car_next, omega_car_next,
            x_block_next, y_block_next, theta_block_next,
            vx_block_next, vy_block_next, omega_block_next
        ], dim=1)
        
        return next_states
    
    def running_cost(self, states, actions):
        """
        Cost function for MPPI - aligned with SSI MPC weights.
        
        Weights match SSI MPC:
        - Q_block_pos = 10.0 (block position tracking)
        - Q_block_theta = 5.0 (block heading tracking)
        - Q_vel = 0.1 (velocity smoothness)
        - Q_contact = 8.0 (contact maintenance/alignment)
        - R_steering = 0.5 (steering effort)
        - R_accel = 0.1 (acceleration effort)
        """
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32)
        
        # Extract block states (indices 5, 6, 7 for position and heading)
        x_block = states[:, 5]
        y_block = states[:, 6]
        theta_block = states[:, 7]
        
        traj_ref = self.trajectory[self.index]
        if isinstance(traj_ref, np.ndarray):
            traj_ref = torch.tensor(traj_ref, dtype=torch.float32, device=states.device)
        
        angle_diff = theta_block - traj_ref[2]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        
        # 1. Block position tracking cost (aligned with SSI MPC: Q_block_pos = 10.0)
        Q_block_pos = 10.0
        position_cost_x = Q_block_pos * (x_block - traj_ref[0]) ** 2
        position_cost_y = Q_block_pos * (y_block - traj_ref[1]) ** 2
        
        # 2. Block heading tracking cost (aligned with SSI MPC: Q_block_theta = 5.0)
        Q_block_theta = 5.0
        heading_cost = Q_block_theta * (angle_diff) ** 2
        
        target_cost = position_cost_x + position_cost_y + heading_cost
        
        # 3. Velocity smoothness (aligned with SSI MPC: Q_vel = 0.1)
        Q_vel = 0.1
        vx_block = states[:, 8]  # Block velocity x
        vy_block = states[:, 9]   # Block velocity y
        velocity_smoothness_cost = Q_vel * (vx_block ** 2 + vy_block ** 2)
        
        # 4. Control effort (aligned with SSI MPC: R_steering = 0.5, R_accel = 0.1)
        R_steering = 5.0
        R_accel = 5.0
        action_cost_steering = R_steering * actions[:, 0] ** 2
        action_cost_accel = R_accel * actions[:, 1] ** 2
        action_cost = action_cost_steering + action_cost_accel
        
        # 5. Contact maintenance/alignment cost (aligned with SSI MPC: Q_contact = 8.0)
        Q_contact = 8.0
        x_car = states[:, 0]
        y_car = states[:, 1]
        theta_car = states[:, 2]
        offset = self.car.offset_to_front
        
        # Bumper position
        bumper_x = x_car + offset * torch.cos(theta_car)
        bumper_y = y_car + offset * torch.sin(theta_car)
        
        # Contact maintenance cost (block at bumper)
        contact_cost_x = Q_contact * (x_block - bumper_x) ** 2
        contact_cost_y = Q_contact * (y_block - bumper_y) ** 2
        contact_cost = contact_cost_x + contact_cost_y
        
        cost = target_cost + velocity_smoothness_cost + action_cost + contact_cost
        
        return cost
    
    def terminal_state_cost(self, s, a):
        """
        Terminal cost for MPPI - aligned with SSI MPC weights.
        
        SSI MPC uses: 5.0 * Q_block_pos and 5.0 * Q_block_theta for terminal cost.
        """
        final_states = s[0, :, -1, :]
        x_block_final = final_states[:, 5]
        y_block_final = final_states[:, 6]
        theta_block_final = final_states[:, 7]
        
        goal_position = self.trajectory[-1, :2]
        if isinstance(goal_position, np.ndarray):
            goal_position = torch.tensor(goal_position, dtype=torch.float32, device=s.device)
        goal_heading = self.trajectory[-1, 2]
        
        # Terminal cost weights (aligned with SSI MPC: 5.0 * Q_block_pos, 5.0 * Q_block_theta)
        Q_block_pos = 10.0
        Q_block_theta = 5.0
        terminal_Q_pos = 5.0 * Q_block_pos
        terminal_Q_theta = 5.0 * Q_block_theta
        
        position_cost = terminal_Q_pos * ((x_block_final - goal_position[0]) ** 2 + 
                                          (y_block_final - goal_position[1]) ** 2)
        
        angle_diff = theta_block_final - goal_heading
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        heading_cost = terminal_Q_theta * (angle_diff) ** 2
        
        return position_cost + heading_cost
    
    def set_trajectory(self, trajectory):
        """Set reference trajectory."""
        self.trajectory = trajectory.cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
        self.index = 0
    
    def get_reference_index(self, obs):
        """Find current reference waypoint (same logic as MPPI)."""
        # obs is 11D state, block position is at indices 5, 6
        block_pos = obs[5:7]
        block_pos = np.array(block_pos)
        
        diff = self.trajectory[:, :2] - block_pos[:2]
        dist = np.linalg.norm(diff[:, :2], axis=1)
        index = dist.argmin()
        
        waypoint_lookahead = 0.16
        while (dist[index] < waypoint_lookahead and index <= len(self.trajectory) - 2):
            index += 1
            index = min(index, len(self.trajectory) - 1)
        
        self.index = index
        return index
    
    def command(self, obs):
        """
        Compute control command with SSI update.
        
        Args:
            obs: Raw observation from gym (quaternion format)
            
        Returns:
            action: [steering, acceleration]
        """
        # Convert gym observation to 11D state
        from scipy.spatial.transform import Rotation as R
        def pose_quat2euler(pose):
            return np.array([pose[0], pose[1], 
                           (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)])
        
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        # Convert to 11D state: [x_car, y_car, theta_car, v_car, omega_car, 
        #                        x_block, y_block, theta_block, vx_block, vy_block, omega_block]
        # For velocities, we need to estimate from previous state or use defaults
        if not hasattr(self, 'state_11d'):
            self.state_11d = None
            self.v_car_last = 0.15  # Initial velocity estimate
            self.omega_car_last = 0.0
            self.vx_block_last = 0.0
            self.vy_block_last = 0.0
            self.omega_block_last = 0.0
        
        # Estimate velocities from position changes if we have previous state
        if self.state_11d is not None:
            dt_est = 0.01  # Typical control timestep
            v_car_est = np.linalg.norm([car_euler[0] - self.state_11d[0], 
                                        car_euler[1] - self.state_11d[1]]) / dt_est
            # Estimate angular velocity from heading change
            theta_diff = car_euler[2] - self.state_11d[2]
            theta_diff = ((theta_diff + np.pi) % (2 * np.pi)) - np.pi
            omega_car_est = theta_diff / dt_est
            
            # Block velocities
            vx_block_est = (block_euler[0] - self.state_11d[5]) / dt_est
            vy_block_est = (block_euler[1] - self.state_11d[6]) / dt_est
            theta_block_diff = block_euler[2] - self.state_11d[7]
            theta_block_diff = ((theta_block_diff + np.pi) % (2 * np.pi)) - np.pi
            omega_block_est = theta_block_diff / dt_est
        else:
            v_car_est = self.v_car_last
            omega_car_est = self.omega_car_last
            vx_block_est = self.vx_block_last
            vy_block_est = self.vy_block_last
            omega_block_est = self.omega_block_last
        
        state_11d = np.array([
            car_euler[0], car_euler[1], car_euler[2], v_car_est, omega_car_est,
            block_euler[0], block_euler[1], block_euler[2],
            vx_block_est, vy_block_est, omega_block_est
        ])
        
        # SSI update
        if self.state_11d is not None:
            self.ssi_update(dt=0.01, x_now=state_11d)
        
        # Store current state and velocities for next iteration
        self.state_11d = state_11d.copy()
        self.v_car_last = v_car_est
        self.omega_car_last = omega_car_est
        self.vx_block_last = vx_block_est
        self.vy_block_last = vy_block_est
        self.omega_block_last = omega_block_est
        
        # MPPI command
        # MPPI's command method expects the state in the format our dynamics function uses (11D)
        # Pass the 11D state directly - MPPI will convert it to torch internally
        action = self.ctrl.command(state_11d)
        
        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
        
        # Store action for SSI update (in acceleration format)
        self.u_last = action_np.copy()
        
        # Convert acceleration to velocity for gym environment
        # Gym expects [steering, velocity], but MPPI outputs [steering, acceleration]
        # Integrate acceleration to get velocity: v = v_prev + a * dt
        dt_gym = 0.01  # Gym timestep
        if not hasattr(self, 'v_car_current'):
            self.v_car_current = 0.15  # Initial velocity
        
        # Update velocity from acceleration
        self.v_car_current = max(0.0, self.v_car_current + action_np[1] * dt_gym)
        self.v_car_current = min(self.car.max_velocity, self.v_car_current)
        
        # Return [steering, velocity] for gym (same format as SSI MPC output)
        action_gym = np.array([action_np[0], self.v_car_current])
        
        return action_gym
