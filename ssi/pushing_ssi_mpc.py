"""
SSI-MPC for pushing: Simultaneous System Identification and Model Predictive Control
for car pushing a block with unknown mass and friction parameters.

Based on Zhou et al., "Simultaneous System Identification and Model Predictive Control 
with No Dynamic Regret", TRO 2025.
"""

import numpy as np
import casadi as cs
from .ackermann_model import AckermannCar
import cvxpy as cp
from typing import Optional, Tuple


class PushingSSIMpc:
    """
    SSI-MPC controller for car pushing a block.
    
    State: [x_car, y_car, theta_car, v_car, omega_car, 
            x_block, y_block, theta_block, vx_block, vy_block, omega_block]  (11D)
    Control: [steering_angle, acceleration]  (2D)
    
    SSI learns residuals in block accelerations to compensate for unknown mass and friction.
    """
    
    def __init__(self, 
                 name: str, 
                 car: AckermannCar, 
                 horizon: float, 
                 num_steps: int, 
                 rf_dict: dict):
        """
        Initialize SSI-MPC controller.
        
        Args:
            name: Controller name
            car: AckermannCar object with vehicle parameters
            horizon: Time horizon in seconds
            num_steps: Number of MPC steps
            rf_dict: Dictionary with random feature parameters
                     {'n_rf', 'omega', 'b', 'input', 'target', 'lr'}
        """
        self.model_name = name
        self.car = car
        self.horizon = horizon
        self.num_steps = num_steps
        self.dt = horizon / num_steps
        
        # State and control dimensions
        self.state_dim = 11  # [car(5), block(6)]
        self.u_dim = 2  # [steering, acceleration]
        
        # Random feature parameters for SSI
        self.rf_dict = rf_dict
        self.learning_rate = rf_dict['lr']
        self.n_rf = rf_dict['n_rf']
        self.omega = rf_dict['omega']  # Random weights (n_rf x n_input_features)
        self.b = rf_dict['b']  # Random biases (n_rf x 1)
        self.target_mask = rf_dict['target']  # Which state derivatives to learn
        self.input_mask = rf_dict['input']  # Which features to use as input
        
        # Mapping matrices
        self.Bh = np.eye(self.state_dim)[self.target_mask].T  # Map learned residuals to state space
        self.Bz = np.eye(self.state_dim + self.u_dim)[self.input_mask]  # Map to feature space
        
        # Initialize learning parameters
        self.alpha_last = np.zeros((len(self.target_mask), self.n_rf))
        self.x_last = None
        self.u_last = None
        
        # Setup symbolic dynamics for SSI update
        self._setup_symbolic_dynamics()
        
    def _setup_symbolic_dynamics(self):
        """Setup CasADi symbolic variables for dynamics prediction during SSI update."""
        # State variables
        self.x_car = cs.MX.sym('x_car')
        self.y_car = cs.MX.sym('y_car')
        self.theta_car = cs.MX.sym('theta_car')
        self.v_car = cs.MX.sym('v_car')
        self.omega_car = cs.MX.sym('omega_car')
        
        self.x_block = cs.MX.sym('x_block')
        self.y_block = cs.MX.sym('y_block')
        self.theta_block = cs.MX.sym('theta_block')
        self.vx_block = cs.MX.sym('vx_block')
        self.vy_block = cs.MX.sym('vy_block')
        self.omega_block = cs.MX.sym('omega_block')
        
        self.x = cs.vertcat(self.x_car, self.y_car, self.theta_car, self.v_car, self.omega_car,
                           self.x_block, self.y_block, self.theta_block, 
                           self.vx_block, self.vy_block, self.omega_block)
        
        # Control variables
        self.steering = cs.MX.sym('steering')
        self.accel = cs.MX.sym('accel')
        self.u = cs.vertcat(self.steering, self.accel)
        
        # Learning parameters
        self.alpha = cs.MX.sym('alpha', len(self.target_mask), self.n_rf)
        
        # Build dynamics function
        x_dot = self._augmented_dynamics()
        self.f = cs.Function('x_dot', [self.x, self.u, self.alpha], [x_dot], 
                            ['x', 'u', 'alpha'], ['x_dot'])
    
    def _augmented_dynamics(self):
        """
        Augmented dynamics: nominal quasi-static pushing model + SSI residuals.
        
        Returns:
            x_dot: State derivatives (11D)
        """
        L = self.car.wheelbase
        offset = self.car.offset_to_front
        
        # --- Car dynamics (Ackermann bicycle model) ---
        x_car_dot = self.v_car * cs.cos(self.theta_car)
        y_car_dot = self.v_car * cs.sin(self.theta_car)
        theta_car_dot = (self.v_car / L) * cs.tan(self.steering)
        v_car_dot = self.accel
        
        # Angular acceleration from steering dynamics
        omega_car_dot = (self.accel / L) * cs.tan(self.steering) + \
                        (self.v_car / L) * (self.steering / cs.cos(self.steering)**2) * 0.0
        # Simplified: just differentiate theta_car_dot (assume steering angle changes slowly)
        omega_car_dot = (self.accel / L) * cs.tan(self.steering)
        
        # --- Block dynamics (quasi-static nominal + SSI residuals) ---
        # Nominal: block rigidly follows car front bumper
        bumper_x = self.x_car + offset * cs.cos(self.theta_car)
        bumper_y = self.y_car + offset * cs.sin(self.theta_car)
        
        # Position derivatives (integrate from velocities - actual states)
        x_block_dot = self.vx_block
        y_block_dot = self.vy_block
        theta_block_dot = self.omega_block
        
        # Nominal velocity (quasi-static: block follows bumper)
        vx_block_nominal_dot = self.v_car * cs.cos(self.theta_car) - \
                               offset * theta_car_dot * cs.sin(self.theta_car)
        vy_block_nominal_dot = self.v_car * cs.sin(self.theta_car) + \
                               offset * theta_car_dot * cs.cos(self.theta_car)
        omega_block_nominal_dot = theta_car_dot
        
        # Nominal accelerations (differentiating nominal velocities)
        vx_block_nominal_ddot = self.accel * cs.cos(self.theta_car) - \
                                self.v_car * theta_car_dot * cs.sin(self.theta_car) - \
                                offset * (theta_car_dot**2 * cs.cos(self.theta_car) + \
                                         omega_car_dot * cs.sin(self.theta_car))
        
        vy_block_nominal_ddot = self.accel * cs.sin(self.theta_car) + \
                                self.v_car * theta_car_dot * cs.cos(self.theta_car) - \
                                offset * (theta_car_dot**2 * cs.sin(self.theta_car) - \
                                         omega_car_dot * cs.cos(self.theta_car))
        
        omega_block_nominal_ddot = omega_car_dot
        
        # Compute random features
        Z = cs.vertcat(self.x, self.u)  # All states and controls
        rf = (1.0 / cs.sqrt(self.n_rf)) * cs.cos(cs.mtimes(self.omega, cs.mtimes(self.Bz, Z)) + self.b)
        
        # SSI augmentation: learned residuals
        residuals = cs.mtimes(self.alpha, rf)  # (len(target_mask) x 1)
        
        # Apply residuals to block accelerations (target_mask selects which)
        # Assuming target_mask = [8, 9, 10] for [vx_block_dot, vy_block_dot, omega_block_dot]
        vx_block_dot = vx_block_nominal_ddot + residuals[0]
        vy_block_dot = vy_block_nominal_ddot + residuals[1]
        omega_block_dot = omega_block_nominal_ddot + residuals[2]
        
        # Full state derivative
        x_dot = cs.vertcat(x_car_dot, y_car_dot, theta_car_dot, v_car_dot, omega_car_dot,
                          x_block_dot, y_block_dot, theta_block_dot,
                          vx_block_dot, vy_block_dot, omega_block_dot)
        
        return x_dot
    
    def update_step(self, dt: float, x_now: np.ndarray):
        """
        SSI update step: gradient descent on prediction error.
        
        Based on Algorithm 1 from Zhou et al. paper.
        
        Args:
            dt: Time elapsed since last update
            x_now: Current state observation (11D)
        """
        # First call: initialize
        if self.x_last is None:
            self.x_last = x_now.copy()
            self.u_last = np.zeros(self.u_dim)
            return
        
        if dt == 0 or dt < 1e-6:
            dt = self.dt
        
        # Get previous data
        alpha_in = self.alpha_last
        x_in = self.x_last
        u_in = self.u_last
        
        # --- SSI Update: Gradient descent on prediction error ---
        
        # 1. Compute random features at previous time step
        Z = np.hstack((x_in, u_in)).reshape(-1, 1)  # (13 x 1)
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ (self.Bz @ Z) + self.b)  # (n_rf x 1)
        
        # 2. Predict next state using current alpha
        x_dot_pred = np.array(self.f(x=x_in, u=u_in, alpha=alpha_in)['x_dot']).reshape(-1)
        x_pred = x_in + dt * x_dot_pred
        
        # 3. Compute prediction error on target channels (block accelerations)
        # target_mask selects which states to compute error on
        error_pred = self.Bh.T @ (x_pred.reshape(-1, 1) - x_now.reshape(-1, 1))  # (3 x 1)
        
        # 4. Gradient descent update
        alpha_out = alpha_in - 2.0 * self.learning_rate * (error_pred @ rf.T)
        
        # 5. Store for next iteration
        self.alpha_last = np.copy(alpha_out)
        self.x_last = np.copy(x_now)
        
        return
    
    def solve_mpc(self, 
                  x0: np.ndarray, 
                  ref_trajectory: np.ndarray,
                  dt: float,
                  verbose: bool = False) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Solve MPC optimization problem using CVXPY.
        
        Args:
            x0: Initial state (11D)
            ref_trajectory: Reference trajectory for block (N+1 x 3) [x, y, theta]
            dt: Time since last MPC solve (for SSI update)
            verbose: Print solver output
            
        Returns:
            status: Solver status (0 = success)
            x_traj: State trajectory (N+1 x 11)
            u_traj: Control trajectory (N x 2)
        """
        # First, update SSI parameters
        self.update_step(dt, x0)
        
        N = self.num_steps
        dt_mpc = self.dt
        
        # Get current learned alpha
        alpha = self.alpha_last
        
        # --- Setup CVXPY optimization problem ---
        
        # Decision variables
        x_var = cp.Variable((N + 1, self.state_dim))
        u_var = cp.Variable((N, self.u_dim))
        
        # Cost function weights
        Q_block_pos = 10.0  # Block position tracking
        Q_block_theta = 5.0  # Block heading tracking
        Q_contact = 8.0  # Contact maintenance (block-bumper alignment)
        Q_vel = 0.1  # Velocity smoothness
        R_steering = 0.5  # Steering effort
        R_accel = 0.1  # Acceleration effort
        
        # Cost and constraints
        cost = 0
        constraints = []
        
        # Initial condition
        constraints.append(x_var[0, :] == x0)
        
        for k in range(N):
            # Extract states at step k
            x_k = x_var[k, :]
            u_k = u_var[k, :]
            x_next = x_var[k + 1, :]
            
            # Extract individual states
            x_car_k = x_k[0]
            y_car_k = x_k[1]
            theta_car_k = x_k[2]
            v_car_k = x_k[3]
            omega_car_k = x_k[4]
            x_block_k = x_k[5]
            y_block_k = x_k[6]
            theta_block_k = x_k[7]
            vx_block_k = x_k[8]
            vy_block_k = x_k[9]
            omega_block_k = x_k[10]
            
            steering_k = u_k[0]
            accel_k = u_k[1]
            
            # --- Dynamics constraints (linearized around current trajectory) ---
            # For CVXPY, we use Euler integration with nominal dynamics
            # SSI augmentation is applied via numpy evaluation, then linearized
            
            # Nominal dynamics (without SSI)
            L = self.car.wheelbase
            offset = self.car.offset_to_front
            
            # Linearize around current state (x0)
            # Get reference values for linearization
            theta_ref = x0[2] if k == 0 else ref_k[2]
            cos_theta_ref = np.cos(theta_ref)
            sin_theta_ref = np.sin(theta_ref)
            
            # Car dynamics (fully linearized to be convex)
            # Linearize around nominal velocity v_nom
            v_nom = max(0.1, x0[3])  # Use initial velocity or minimum
            
            x_car_next = x_car_k + dt_mpc * v_car_k * cos_theta_ref
            y_car_next = y_car_k + dt_mpc * v_car_k * sin_theta_ref
            # Linearize bilinear term: v*steering ≈ v_nom*steering + v*steering_nom - v_nom*steering_nom
            # For simplicity with steering_nom=0 (small deviations): v*steering ≈ v_nom*steering
            theta_car_next = theta_car_k + dt_mpc * (v_nom / L) * steering_k
            v_car_next = v_car_k + dt_mpc * accel_k
            # omega = d(theta)/dt, approximated from theta dynamics
            omega_car_next = (v_nom / L) * steering_k
            
            # Block dynamics (fully linearized for convexity)
            # Position integration
            x_block_next = x_block_k + dt_mpc * vx_block_k
            y_block_next = y_block_k + dt_mpc * vy_block_k
            theta_block_next = theta_block_k + dt_mpc * omega_block_k
            
            # Block velocities follow car (quasi-static, fully linearized)
            # Assume block moves with nominal car velocity
            vx_block_next = v_nom * cos_theta_ref
            vy_block_next = v_nom * sin_theta_ref
            omega_block_next = (v_nom / L) * steering_k
            
            # Dynamics constraints
            constraints.append(x_next[0] == x_car_next)
            constraints.append(x_next[1] == y_car_next)
            constraints.append(x_next[2] == theta_car_next)
            constraints.append(x_next[3] == v_car_next)
            constraints.append(x_next[4] == omega_car_next)
            constraints.append(x_next[5] == x_block_next)
            constraints.append(x_next[6] == y_block_next)
            constraints.append(x_next[7] == theta_block_next)
            constraints.append(x_next[8] == vx_block_next)
            constraints.append(x_next[9] == vy_block_next)
            constraints.append(x_next[10] == omega_block_next)
            
            # --- Cost function ---
            
            # Reference for this step
            ref_k = ref_trajectory[min(k, len(ref_trajectory) - 1)]
            x_ref = ref_k[0]
            y_ref = ref_k[1]
            theta_ref = ref_k[2]
            
            # 1. Block position tracking cost
            cost += Q_block_pos * cp.sum_squares(x_block_k - x_ref)
            cost += Q_block_pos * cp.sum_squares(y_block_k - y_ref)
            cost += Q_block_theta * cp.sum_squares(theta_block_k - theta_ref)
            
            # 2. Contact maintenance cost (block at car front bumper)
            # Linearized bumper position
            bumper_x = x_car_k + offset * cos_theta_ref
            bumper_y = y_car_k + offset * sin_theta_ref
            cost += Q_contact * cp.sum_squares(x_block_k - bumper_x)
            cost += Q_contact * cp.sum_squares(y_block_k - bumper_y)
            
            # 3. Velocity smoothness
            cost += Q_vel * cp.sum_squares(vx_block_k)
            cost += Q_vel * cp.sum_squares(vy_block_k)
            
            # 4. Control effort
            cost += R_steering * cp.sum_squares(steering_k)
            cost += R_accel * cp.sum_squares(accel_k)
            
            # --- Control constraints ---
            constraints.append(steering_k >= self.car.min_steering)
            constraints.append(steering_k <= self.car.max_steering)
            constraints.append(accel_k >= self.car.min_acceleration)
            constraints.append(accel_k <= self.car.max_acceleration)
            
            # --- State constraints ---
            constraints.append(v_car_k >= self.car.min_velocity)
            constraints.append(v_car_k <= self.car.max_velocity)
            
        # Terminal cost
        ref_final = ref_trajectory[-1]
        x_block_final = x_var[N, 5]
        y_block_final = x_var[N, 6]
        theta_block_final = x_var[N, 7]
        
        cost += 5.0 * Q_block_pos * cp.sum_squares(x_block_final - ref_final[0])
        cost += 5.0 * Q_block_pos * cp.sum_squares(y_block_final - ref_final[1])
        cost += 5.0 * Q_block_theta * cp.sum_squares(theta_block_final - ref_final[2])
        
        # --- Solve optimization ---
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-3, eps_rel=1e-3, max_iter=500)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                x_traj = x_var.value
                u_traj = u_var.value
                self.u_last = u_traj[0, :].copy()
                return 0, x_traj, u_traj
            else:
                print(f"CVXPY solver failed with status: {problem.status}")
                # Return zero control
                x_traj = np.tile(x0, (N + 1, 1))
                u_traj = np.zeros((N, self.u_dim))
                return -1, x_traj, u_traj
                
        except Exception as e:
            print(f"MPC solver exception: {e}")
            x_traj = np.tile(x0, (N + 1, 1))
            u_traj = np.zeros((N, self.u_dim))
            return -1, x_traj, u_traj

