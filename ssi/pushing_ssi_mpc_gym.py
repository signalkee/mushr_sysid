"""
Gym integration for SSI-MPC pushing controller.

Replaces MPPI-based PushingController with SSI-MPC.
"""

import numpy as np
import torch
from .pushing_ssi_mpc import PushingSSIMpc
from .ackermann_model import AckermannCar
from scipy.spatial.transform import Rotation as R


class PushingSSIMPCController:
    """
    SSI-MPC controller for pushing tasks in gym environment.
    
    Integrates with MushrBlockEnv from mushr_mujoco_gym.
    """
    
    def __init__(self, 
                 horizon: float = 0.3,
                 num_steps: int = 30,
                 n_rf: int = 50,
                 lr: float = 0.15,
                 kernel_std: float = 0.3):
        """
        Initialize SSI-MPC pushing controller.
        
        Args:
            horizon: MPC time horizon (seconds)
            num_steps: Number of MPC steps
            n_rf: Number of random features for SSI
            lr: Learning rate for SSI gradient descent
            kernel_std: Gaussian kernel standard deviation
        """
        self.horizon = horizon
        self.num_steps = num_steps
        self.n_rf = n_rf
        self.lr = lr
        self.kernel_std = kernel_std
        
        # Initialize car model
        self.car = AckermannCar()
        
        # Reference trajectory
        self.trajectory = None
        self.trajectory_index = 0
        self.reached_goal = False
        self.waypoint_lookahead = 0.16  # meters
        self.threshold = 0.08  # meters (goal tolerance)
        
        # Timing
        self.last_solve_time = None
        
        # Setup random features and MPC
        self._setup_random_features()
        self._setup_mpc()
        
    def _setup_random_features(self):
        """Initialize random features for SSI learning."""
        # Input mask: which features to use for learning
        # We use all 11 states as input features (no control inputs in features)
        self.input_mask = list(range(11))  # All states [0-10]
        
        # Target mask: which state derivatives to learn
        # We learn residuals in block accelerations: [vx_block_dot, vy_block_dot, omega_block_dot]
        self.target_mask = [8, 9, 10]  # Indices of block velocity states
        
        # Draw random features from Gaussian kernel
        # omega ~ N(0, kernel_std^2)
        # b ~ Uniform[0, 2*pi]
        self.omega = np.random.normal(0.0, self.kernel_std, 
                                     (self.n_rf, len(self.input_mask)))
        self.b = np.random.uniform(0.0, 2.0 * np.pi, (self.n_rf, 1))
        
        # Package into dictionary
        self.rf_dict = {
            'n_rf': self.n_rf,
            'omega': self.omega,
            'b': self.b,
            'input': self.input_mask,
            'target': self.target_mask,
            'lr': self.lr
        }
        
    def _setup_mpc(self):
        """Initialize MPC solver."""
        self.mpc_solver = PushingSSIMpc(
            name='mushr_pushing',
            car=self.car,
            horizon=self.horizon,
            num_steps=self.num_steps,
            rf_dict=self.rf_dict
        )
        print(f"Initialized SSI-MPC controller with N={self.num_steps}, T={self.horizon}s")
        
    def set_trajectory(self, trajectory: torch.Tensor):
        """
        Set reference trajectory for block to follow.
        
        Args:
            trajectory: Tensor of shape (N, 3) with [x, y, theta] for each waypoint
        """
        # Convert to numpy if needed
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        
        self.trajectory = trajectory
        self.trajectory_index = 0
        self.reached_goal = False
        print(f"Set trajectory with {len(trajectory)} waypoints")
        
    def get_reference_index(self, obs: np.ndarray) -> int:
        """
        Get current reference waypoint index based on block position.
        
        Args:
            obs: Current observation in Euler format [car(3), block(3)]
            
        Returns:
            Current reference index
        """
        if self.trajectory is None:
            return 0
        
        # Extract block position
        block_pos = obs[3:5]  # [x, y]
        
        # Find closest waypoint
        diff = self.trajectory[:, :2] - block_pos[:2]
        dist = np.linalg.norm(diff, axis=1)
        index = dist.argmin()
        
        # Look ahead for next waypoint
        while (dist[index] < self.waypoint_lookahead and 
               index < len(self.trajectory) - 1):
            index += 1
        
        # Check if goal reached
        if np.linalg.norm(self.trajectory[-1, :2] - block_pos[:2]) < self.threshold:
            self.reached_goal = True
            print("Goal reached!")
        
        self.trajectory_index = index
        return index
    
    def _extract_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract 11D state vector from gym observation.
        
        Gym observation format (from MushrBlockEnv):
            obs[0:6]: car pose [x, y, qw, qx, qy, qz]
            obs[6:12]: block pose [x, y, qw, qx, qy, qz]
            obs[12:16]: car velocity [vx, vy, vtheta, ?]
            obs[16:20]: block velocity [vx, vy, vtheta, ?]
        
        State format:
            [x_car, y_car, theta_car, v_car, omega_car,
             x_block, y_block, theta_block, vx_block, vy_block, omega_block]
        
        Args:
            obs: Raw observation from gym environment
            
        Returns:
            state: 11D state vector
        """
        # Extract car pose
        car_x = obs[0]
        car_y = obs[1]
        car_quat = obs[2:6]  # [qw, qx, qy, qz]
        
        # Convert quaternion to Euler angle (yaw)
        car_quat_scipy = np.array([car_quat[1], car_quat[2], car_quat[3], car_quat[0]])  # [qx,qy,qz,qw]
        car_euler = R.from_quat(car_quat_scipy).as_euler('xyz', degrees=False)
        car_theta = car_euler[2]
        
        # Extract block pose
        block_x = obs[6]
        block_y = obs[7]
        block_quat = obs[8:12]  # [qw, qx, qy, qz]
        
        # Convert quaternion to Euler angle (yaw)
        block_quat_scipy = np.array([block_quat[1], block_quat[2], block_quat[3], block_quat[0]])
        block_euler = R.from_quat(block_quat_scipy).as_euler('xyz', degrees=False)
        block_theta = block_euler[2]
        
        # Extract velocities
        car_vx = obs[12]
        car_vy = obs[13]
        car_omega = obs[14]  # angular velocity
        
        block_vx = obs[16]
        block_vy = obs[17]
        block_omega = obs[18]
        
        # Compute car speed (scalar)
        car_v = np.sqrt(car_vx**2 + car_vy**2)
        
        # Assemble state vector
        state = np.array([
            car_x, car_y, car_theta, car_v, car_omega,
            block_x, block_y, block_theta, block_vx, block_vy, block_omega
        ])
        
        return state
    
    def _generate_reference_trajectory(self, current_index: int) -> np.ndarray:
        """
        Generate reference trajectory for MPC horizon.
        
        Args:
            current_index: Current position along reference trajectory
            
        Returns:
            ref_traj: Reference trajectory (N+1 x 3) [x, y, theta]
        """
        if self.trajectory is None:
            return np.zeros((self.num_steps + 1, 3))
        
        N = self.num_steps
        traj_len = len(self.trajectory)
        
        # Sample reference points along trajectory
        ref_traj = np.zeros((N + 1, 3))
        
        for i in range(N + 1):
            idx = min(current_index + i, traj_len - 1)
            ref_traj[i, :] = self.trajectory[idx, :]
        
        return ref_traj
    
    def command(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute control command using SSI-MPC.
        
        Args:
            obs: Current observation from gym environment
            
        Returns:
            action: Control action [steering_angle, velocity]
        """
        import time
        
        # Extract state
        state = self._extract_state(obs)
        
        # Update reference index
        obs_euler = np.concatenate([
            state[:3],   # car [x, y, theta]
            state[5:8]   # block [x, y, theta]
        ])
        ref_idx = self.get_reference_index(obs_euler)
        
        # Generate reference trajectory
        ref_traj = self._generate_reference_trajectory(ref_idx)
        
        # Compute dt for SSI update
        current_time = time.time()
        if self.last_solve_time is None:
            dt = self.mpc_solver.dt
        else:
            dt = current_time - self.last_solve_time
        self.last_solve_time = current_time
        
        # Solve MPC
        status, x_traj, u_traj = self.mpc_solver.solve_mpc(state, ref_traj, dt, verbose=False)
        
        if status != 0:
            print(f"Warning: MPC solver failed with status {status}")
            return np.array([0.0, 0.0])
        
        # Extract first control action
        steering_angle = u_traj[0, 0]
        acceleration = u_traj[0, 1]
        
        # Use MPC's planned next velocity (from trajectory)
        # x_traj contains the full state including v_car at index 3
        # Use the velocity from next timestep (index 1)
        planned_velocity = x_traj[1, 3] if x_traj.shape[0] > 1 else state[3]
        
        # Apply minimum pushing velocity to overcome friction
        # (from pushing_cost_function.py min_push_velocity = 0.2 m/s)
        min_push_vel = 0.15  # Minimum velocity to actually push the block
        if planned_velocity < min_push_vel:
            # If MPC plans slow, boost to minimum pushing velocity
            commanded_velocity = min_push_vel
        else:
            commanded_velocity = planned_velocity
        
        # Clip to valid range
        commanded_velocity = np.clip(commanded_velocity, 
                                    self.car.min_velocity, 
                                    self.car.max_velocity)
        
        action = np.array([steering_angle, commanded_velocity])
        
        return action
    
    def reset(self):
        """Reset controller state."""
        self.trajectory_index = 0
        self.reached_goal = False
        self.last_solve_time = None
        # Note: SSI learning parameters (alpha) are NOT reset to allow transfer learning


def pose_quat2euler(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose from quaternion to Euler representation.
    
    Args:
        pose: [x, y, qw, qx, qy, qz]
        
    Returns:
        [x, y, theta]
    """
    x, y = pose[0], pose[1]
    quat = pose[2:6]  # [qw, qx, qy, qz]
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])  # [qx, qy, qz, qw]
    euler = R.from_quat(quat_scipy).as_euler('xyz', degrees=False)
    theta = euler[2]
    return np.array([x, y, theta])


def pose_euler2quat(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose from Euler to quaternion representation.
    
    Args:
        pose: [x, y, theta]
        
    Returns:
        [x, y, qw, qx, qy, qz]
    """
    x, y, theta = pose[0], pose[1], pose[2]
    quat = R.from_euler('xyz', [0, 0, theta], degrees=False).as_quat()  # [qx, qy, qz, qw]
    return np.array([x, y, quat[3], quat[0], quat[1], quat[2]])  # [x, y, qw, qx, qy, qz]

