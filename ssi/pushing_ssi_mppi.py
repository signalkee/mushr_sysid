"""
SSI-MPPI: Simultaneous System Identification + Model Predictive Path Integral

Combines:
- MPPI: Sampling-based MPC (from path_tracking_controller.py)
- SSI: Online learning of unknown mass/friction

The dynamics model used in MPPI rollouts is augmented with SSI residuals.
"""

import numpy as np
import torch
from pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal


class PushingSSIMPPI:
    """
    SSI-MPPI controller for pushing with online learning.
    
    Augments MPPI dynamics with SSI-learned residuals for mass/friction.
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
        
        # SSI parameters
        self.n_rf = n_rf
        self.lr = lr
        self.kernel_std = kernel_std
        
        # Mass/friction
        self.nominal_mass = nominal_mass
        self.nominal_friction = nominal_friction
        self.true_mass = true_mass
        self.true_friction = true_friction
        
        print(f"   SSI-MPPI Nominal: mass={nominal_mass} kg, friction={nominal_friction}")
        print(f"   SSI-MPPI True:    mass={true_mass} kg, friction={true_friction}")
        
        # Random features (for block velocities: vx, vy)
        self.state_dim = 6  # [car_x, car_y, car_theta, block_x, block_y, block_theta]
        self.input_mask = [0, 1, 2, 3, 4, 5]  # Use all states as features
        self.target_mask = [0, 1]  # Learn residuals for block vx, vy (indices in velocity space)
        
        # Random features
        n_inputs = len(self.input_mask)
        self.omega = np.random.normal(0.0, kernel_std, (n_rf, n_inputs))
        self.b = np.random.uniform(0.0, 2.0 * np.pi, (n_rf, 1))
        
        # Learning parameters
        self.alpha = np.zeros((len(self.target_mask), n_rf))
        self.x_last = None
        self.u_last = None
        
        # MPPI setup
        self.reset_mppi()
        
    def reset_mppi(self):
        """Initialize MPPI controller."""
        TIMESTEPS = 30
        N_SAMPLES = 200
        ACTION_LOW = [-0.17, 0]
        ACTION_HIGH = [0.17, 0.21]
        
        nx = 6  # state dimension
        noise_sigma = torch.tensor([[0.05, 0.0],
                                    [0.0, .09]], device=self.device, dtype=torch.float32)
        lambda_ = 1e-2
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
        Z = np.hstack([states, actions]).reshape(-1, 1)
        # Select features based on input mask
        Z_masked = Z[self.input_mask, :]
        # Compute features
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ Z_masked + self.b)
        return rf
    
    def ssi_update(self, dt, x_now):
        """
        SSI update step: learn residuals in block dynamics.
        
        Args:
            dt: Time since last update
            x_now: Current state (6D)
        """
        if self.x_last is None:
            self.x_last = x_now.copy()
            self.u_last = np.zeros(2)
            return
        
        if dt == 0 or dt < 1e-6:
            dt = 0.01
        
        # Get previous data
        alpha_in = self.alpha
        x_in = self.x_last
        u_in = self.u_last
        
        # Compute random features at previous state
        rf = self.compute_random_features(x_in, u_in)
        
        # Predict next state using nominal dynamics + learned residuals
        x_pred = self.predict_next_state_numpy(x_in, u_in, alpha_in, dt)
        
        # Error in block velocities (we learn corrections to vx, vy)
        # Approximate velocities from positions
        vx_pred = (x_pred[3] - x_in[3]) / dt
        vy_pred = (x_pred[4] - x_in[4]) / dt
        vx_actual = (x_now[3] - x_in[3]) / dt
        vy_actual = (x_now[4] - x_in[4]) / dt
        
        error = np.array([[vx_pred - vx_actual], [vy_pred - vy_actual]])
        
        # Gradient descent update
        alpha_out = alpha_in - 2.0 * self.lr * (error @ rf.T)
        
        # Store
        self.alpha = np.copy(alpha_out)
        self.x_last = np.copy(x_now)
    
    def predict_next_state_numpy(self, states, actions, alpha, dt):
        """Predict next state using nominal dynamics + SSI residuals (numpy)."""
        # Nominal dynamics (from pushing_cost_function.py)
        x_now = states[0]
        y_now = states[1]
        theta_now = states[2]
        block_x = states[3]
        block_y = states[4]
        block_theta = states[5]
        
        steering = actions[0]
        speed = actions[1]
        
        # Car dynamics
        x_dot = speed * np.cos(theta_now)
        y_dot = speed * np.sin(theta_now)
        theta_dot = (speed * np.tan(steering)) / 0.295
        
        # Block follows car (quasi-static)
        offset_x_car = block_x - x_now
        offset_y_car = block_y - y_now
        
        # Transform to car frame
        cos_th = np.cos(theta_now)
        sin_th = np.sin(theta_now)
        offset_x_car_frame = cos_th * offset_x_car + sin_th * offset_y_car
        offset_y_car_frame = -sin_th * offset_x_car + cos_th * offset_y_car
        
        # Next car state
        x_next = x_now + x_dot * dt
        y_next = y_now + y_dot * dt
        theta_next = theta_now + theta_dot * dt
        
        # Block follows (nominal)
        cos_th_next = np.cos(theta_next)
        sin_th_next = np.sin(theta_next)
        block_x_next = x_next + (cos_th_next * offset_x_car_frame - sin_th_next * offset_y_car_frame)
        block_y_next = y_next + (sin_th_next * offset_x_car_frame + cos_th_next * offset_y_car_frame)
        
        # SSI residuals (add learned correction to block motion)
        residuals = alpha @ self.compute_random_features(states, actions)
        block_x_next += residuals[0, 0] * dt
        block_y_next += residuals[1, 0] * dt
        
        block_theta_next = theta_next
        
        return np.array([x_next, y_next, theta_next, block_x_next, block_y_next, block_theta_next])
    
    def push_dynamics_ssi(self, states, actions):
        """
        Push dynamics with SSI augmentation for MPPI rollouts.
        
        Args:
            states: (N_samples, 6) tensor
            actions: (N_samples, 2) tensor
            
        Returns:
            next_states: (N_samples, 6) tensor
        """
        dt = 0.01
        min_push_velocity = 0.2
        
        # Extract states
        x_now = states[:, 0]
        y_now = states[:, 1]
        Th_now = states[:, 2]
        block_x = states[:, 3]
        block_y = states[:, 4]
        block_theta = states[:, 5]
        
        steering_angle = actions[:, 0]
        speed = actions[:, 1]
        
        # Car dynamics
        x_dot = speed * torch.cos(Th_now) * dt
        y_dot = speed * torch.sin(Th_now) * dt
        theta_dot = ((speed * torch.tan(steering_angle)) / 0.295) * dt
        is_pushing = torch.abs(speed) >= min_push_velocity
        
        x_next = x_now + x_dot
        y_next = y_now + y_dot
        Th_next = Th_now + theta_dot
        
        # Block dynamics (quasi-static with SSI)
        dx_global = block_x - x_now
        dy_global = block_y - y_now
        
        cos_th = torch.cos(Th_now)
        sin_th = torch.sin(Th_now)
        
        offset_x_car = cos_th * dx_global + sin_th * dy_global
        offset_y_car = -sin_th * dx_global + cos_th * dy_global
        
        cos_th_next = torch.cos(Th_next)
        sin_th_next = torch.sin(Th_next)
        
        block_x_push = x_next + (cos_th_next * offset_x_car - sin_th_next * offset_y_car)
        block_y_push = y_next + (sin_th_next * offset_x_car + cos_th_next * offset_y_car)
        
        # Add SSI correction (convert alpha to torch)
        alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=states.device)
        
        # Compute random features for batch (simplified - use mean state)
        # For efficiency in MPPI, we use current learned alpha without recomputing features per sample
        # This is an approximation but keeps MPPI fast
        if torch.linalg.norm(alpha_torch) > 1e-6:
            # Apply learned correction (approximate as constant offset)
            correction_x = alpha_torch[0, :].mean() * dt * 0.1
            correction_y = alpha_torch[1, :].mean() * dt * 0.1
            block_x_push += correction_x
            block_y_push += correction_y
        
        heading_offset = block_theta - Th_now
        block_theta_push = Th_next + heading_offset
        
        # Apply based on pushing condition
        block_x_next = torch.where(is_pushing, block_x_push, block_x)
        block_y_next = torch.where(is_pushing, block_y_push, block_y)
        block_theta_next = torch.where(is_pushing, block_theta_push, block_theta)
        
        x_next = torch.where(is_pushing, x_next, x_now)
        y_next = torch.where(is_pushing, y_next, y_now)
        Th_next = torch.where(is_pushing, Th_next, Th_now)
        
        next_states = torch.stack(
            (x_next, y_next, Th_next, block_x_next, block_y_next, block_theta_next),
            dim=1
        )
        
        return next_states
    
    def running_cost(self, states, actions):
        """Cost function for MPPI (same as original)."""
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32)
        
        block = states[:, 3:6]
        traj_ref = self.trajectory[self.index]
        if isinstance(traj_ref, np.ndarray):
            traj_ref = torch.tensor(traj_ref, dtype=torch.float32, device=states.device)
        
        angle_diff = block[:, 2] - traj_ref[2]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        
        position_cost_x = 1 * (block[:, 0] - traj_ref[0]) ** 2
        position_cost_y = 1 * (block[:, 1] - traj_ref[1]) ** 2
        heading_cost = 2 * (angle_diff) ** 2
        
        target_cost = position_cost_x + position_cost_y + heading_cost
        
        action_cost_throttle = 0.001 * actions[:, 0] ** 2
        action_cost_steering = 0.01 * actions[:, 1] ** 2
        action_cost = action_cost_throttle + action_cost_steering
        
        car = states[:, :3]
        car_to_block_angle = torch.atan2(block[:, 1] - car[:, 1], block[:, 0] - car[:, 0])
        alignment_error = car[:, 2] - car_to_block_angle
        alignment_error = ((alignment_error + np.pi) % (2 * np.pi)) - np.pi
        alignment_cost = 0.5 * alignment_error ** 2
        
        cost = target_cost + action_cost + alignment_cost
        
        return cost
    
    def terminal_state_cost(self, s, a):
        """Terminal cost for MPPI."""
        final_states = s[0, :, -1, :]
        final_block = final_states[:, 3:6]
        
        goal_position = self.trajectory[-1, :2]
        if isinstance(goal_position, np.ndarray):
            goal_position = torch.tensor(goal_position, dtype=torch.float32, device=s.device)
        goal_heading = self.trajectory[-1, 2]
        
        position_cost = torch.sum((final_block[:, :2] - goal_position) ** 2, dim=1)
        
        angle_diff = final_block[:, 2] - goal_heading
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        heading_cost = 1.5 * (angle_diff) ** 2
        
        return position_cost + heading_cost
    
    def set_trajectory(self, trajectory):
        """Set reference trajectory."""
        self.trajectory = trajectory.cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
        self.index = 0
    
    def get_reference_index(self, obs):
        """Find current reference waypoint (same logic as MPPI)."""
        block_pose = obs[3:5]
        block_pose = np.array(block_pose)
        
        diff = self.trajectory[:, :2] - block_pose[:2]
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
            obs: Raw observation from gym
            
        Returns:
            action: [steering, velocity]
        """
        # Update SSI (online learning)
        # Note: obs is in quaternion form, need to extract 6D state
        # For simplicity, we track the 6D state ourselves
        if not hasattr(self, 'state_6d'):
            self.state_6d = None
        
        # Convert obs to 6D state (simplified)
        from scipy.spatial.transform import Rotation as R
        def pose_quat2euler(pose):
            return np.array([pose[0], pose[1], 
                           (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)])
        
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        state_6d = np.array([car_euler[0], car_euler[1], car_euler[2],
                            block_euler[0], block_euler[1], block_euler[2]])
        
        # SSI update
        if self.state_6d is not None:
            self.ssi_update(dt=0.01, x_now=state_6d)
        self.state_6d = state_6d
        
        # MPPI command
        action = self.ctrl.command(obs)
        self.u_last = action  # Store for SSI
        
        return action

