"""
SSI-MPC for pushing with NONLINEAR dynamics and mass/friction learning.

Uses CasADi with IPOPT solver (no linearization, proper physics).
"""

import numpy as np
import casadi as cs
from .ackermann_model import AckermannCar
from typing import Tuple


class PushingSSIMpcNonlinear:
    """
    SSI-MPC with nonlinear dynamics and physics-based mass/friction.
    
    State: [x_car, y_car, theta_car, v_car, omega_car, 
            x_block, y_block, theta_block, vx_block, vy_block, omega_block]  (11D)
    Control: [steering_angle, acceleration]  (2D)
    
    SSI learns residuals to compensate for WRONG nominal mass/friction.
    """
    
    def __init__(self, 
                 name: str, 
                 car: AckermannCar, 
                 horizon: float, 
                 num_steps: int, 
                 rf_dict: dict,
                 true_mass: float = None,
                 true_friction: float = None):
        """
        Initialize nonlinear SSI-MPC.
        
        Args:
            name: Controller name
            car: AckermannCar with NOMINAL (wrong) mass/friction
            horizon: MPC time horizon
            num_steps: Number of MPC steps
            rf_dict: Random feature parameters
            true_mass: Actual block mass (if None, use nominal)
            true_friction: Actual friction (if None, use nominal)
        """
        self.model_name = name
        self.car = car
        self.horizon = horizon
        self.num_steps = num_steps
        self.dt = horizon / num_steps
        
        self.state_dim = 11
        self.u_dim = 2
        
        # SSI parameters
        self.rf_dict = rf_dict
        self.learning_rate = rf_dict['lr']
        self.n_rf = rf_dict['n_rf']
        self.omega = rf_dict['omega']
        self.b = rf_dict['b']
        self.target_mask = rf_dict['target']
        self.input_mask = rf_dict['input']
        
        # Mapping matrices
        self.Bh = np.eye(self.state_dim)[self.target_mask].T
        self.Bz = np.eye(self.state_dim + self.u_dim)[self.input_mask]
        
        # Learning parameters
        self.alpha_last = np.zeros((len(self.target_mask), self.n_rf))
        self.x_last = None
        self.u_last = None
        
        # True vs nominal parameters (for testing SSI)
        self.true_mass = true_mass if true_mass else car.block_mass_nominal
        self.true_friction = true_friction if true_friction else car.block_friction_nominal
        
        print(f"   Nominal mass: {car.block_mass_nominal:.2f} kg, friction: {car.block_friction_nominal:.2f}")
        print(f"   True mass:    {self.true_mass:.2f} kg, friction: {self.true_friction:.2f}")
        
        # Setup symbolic dynamics
        self._setup_symbolic_dynamics()
        
    def _setup_symbolic_dynamics(self):
        """Setup CasADi symbolic dynamics with mass/friction."""
        # States
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
        
        # Controls
        self.steering = cs.MX.sym('steering')
        self.accel = cs.MX.sym('accel')
        self.u = cs.vertcat(self.steering, self.accel)
        
        # Learning parameters
        self.alpha = cs.MX.sym('alpha', len(self.target_mask), self.n_rf)
        
        # Build dynamics
        x_dot = self._physics_based_dynamics()
        self.f = cs.Function('x_dot', [self.x, self.u, self.alpha], [x_dot], 
                            ['x', 'u', 'alpha'], ['x_dot'])
    
    def _physics_based_dynamics(self):
        """
        Physics-based dynamics with mass and friction.
        Uses NOMINAL (potentially wrong) parameters + SSI residuals.
        """
        L = self.car.wheelbase
        offset = self.car.offset_to_front
        
        # Use NOMINAL (wrong) mass and friction
        m_block = self.car.block_mass_nominal
        mu = self.car.block_friction_nominal
        g = self.car.gravity
        
        # --- Car dynamics (Ackermann) ---
        x_car_dot = self.v_car * cs.cos(self.theta_car)
        y_car_dot = self.v_car * cs.sin(self.theta_car)
        theta_car_dot = (self.v_car / L) * cs.tan(self.steering)
        v_car_dot = self.accel
        omega_car_dot = (self.accel / L) * cs.tan(self.steering)
        
        # --- Block dynamics with PHYSICS ---
        # Position integration
        x_block_dot = self.vx_block
        y_block_dot = self.vy_block
        theta_block_dot = self.omega_block
        
        # Contact force from car (push force)
        # Compute vector from car to block
        dx = self.x_block - self.x_car
        dy = self.y_block - self.y_car
        dist = cs.sqrt(dx**2 + dy**2 + 1e-6)
        
        # Desired position (block at car bumper)
        bumper_x = self.x_car + offset * cs.cos(self.theta_car)
        bumper_y = self.y_car + offset * cs.sin(self.theta_car)
        
        # Error in position
        error_x = self.x_block - bumper_x
        error_y = self.y_block - bumper_y
        
        # Virtual spring force (pushes block to bumper position)
        k_spring = 50.0  # Spring stiffness
        F_contact_x = -k_spring * error_x
        F_contact_y = -k_spring * error_y
        
        # Friction force (opposes motion)
        v_block_norm = cs.sqrt(self.vx_block**2 + self.vy_block**2 + 1e-6)
        F_friction_x = -mu * m_block * g * (self.vx_block / v_block_norm)
        F_friction_y = -mu * m_block * g * (self.vy_block / v_block_norm)
        
        # Newton's law with NOMINAL mass (F = ma)
        vx_block_ddot_nominal = (F_contact_x + F_friction_x) / m_block
        vy_block_ddot_nominal = (F_contact_y + F_friction_y) / m_block
        omega_block_ddot_nominal = 0.0  # Simplified (no torque)
        
        # Compute random features
        Z = cs.vertcat(self.x, self.u)
        rf = (1.0 / cs.sqrt(self.n_rf)) * cs.cos(cs.mtimes(self.omega, cs.mtimes(self.Bz, Z)) + self.b)
        
        # SSI augmentation: learned residuals
        residuals = cs.mtimes(self.alpha, rf)  # (3 x 1)
        
        # Apply residuals to block accelerations
        vx_block_ddot = vx_block_ddot_nominal + residuals[0]
        vy_block_ddot = vy_block_ddot_nominal + residuals[1]
        omega_block_ddot = omega_block_ddot_nominal + residuals[2]
        
        # Full state derivative
        x_dot = cs.vertcat(x_car_dot, y_car_dot, theta_car_dot, v_car_dot, omega_car_dot,
                          x_block_dot, y_block_dot, theta_block_dot,
                          vx_block_ddot, vy_block_ddot, omega_block_ddot)
        
        return x_dot
    
    def update_step(self, dt: float, x_now: np.ndarray):
        """SSI update step (same as before)."""
        if self.x_last is None:
            self.x_last = x_now.copy()
            self.u_last = np.zeros(self.u_dim)
            return
        
        if dt == 0 or dt < 1e-6:
            dt = self.dt
        
        alpha_in = self.alpha_last
        x_in = self.x_last
        u_in = self.u_last
        
        # Random features
        Z = np.hstack((x_in, u_in)).reshape(-1, 1)
        rf = (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ (self.Bz @ Z) + self.b)
        
        # Predict
        x_dot_pred = np.array(self.f(x=x_in, u=u_in, alpha=alpha_in)['x_dot']).reshape(-1)
        x_pred = x_in + dt * x_dot_pred
        
        # Error
        error_pred = self.Bh.T @ (x_pred.reshape(-1, 1) - x_now.reshape(-1, 1))
        
        # Update
        alpha_out = alpha_in - 2.0 * self.learning_rate * (error_pred @ rf.T)
        
        self.alpha_last = np.copy(alpha_out)
        self.x_last = np.copy(x_now)
    
    def solve_mpc(self, 
                  x0: np.ndarray, 
                  ref_trajectory: np.ndarray,
                  dt: float,
                  verbose: bool = False) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Solve MPC using CasADi Opti with IPOPT (nonlinear).
        """
        # Update SSI
        self.update_step(dt, x0)
        
        N = self.num_steps
        dt_mpc = self.dt
        alpha = self.alpha_last
        
        # Create CasADi Opti problem
        opti = cs.Opti()
        
        # Decision variables
        X = opti.variable(self.state_dim, N + 1)
        U = opti.variable(self.u_dim, N)
        
        # Cost weights
        Q_pos = 5.0
        Q_theta = 10.0
        Q_contact = 12.0
        R_steering = 0.5
        R_accel = 0.1
        
        cost = 0
        
        # Initial condition
        opti.subject_to(X[:, 0] == x0)
        
        for k in range(N):
            # Current state and control
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            
            # Dynamics constraint (Euler integration, NONLINEAR)
            x_dot_k = self.f(x=x_k, u=u_k, alpha=alpha)['x_dot']
            opti.subject_to(x_next == x_k + dt_mpc * x_dot_k)
            
            # Reference
            ref_k = ref_trajectory[min(k, len(ref_trajectory) - 1)]
            
            # Cost
            # Block tracking
            cost += Q_pos * (x_k[5] - ref_k[0])**2  # x_block
            cost += Q_pos * (x_k[6] - ref_k[1])**2  # y_block
            cost += Q_theta * (x_k[7] - ref_k[2])**2  # theta_block
            
            # Contact maintenance
            bumper_x = x_k[0] + self.car.offset_to_front * cs.cos(x_k[2])
            bumper_y = x_k[1] + self.car.offset_to_front * cs.sin(x_k[2])
            cost += Q_contact * (x_k[5] - bumper_x)**2
            cost += Q_contact * (x_k[6] - bumper_y)**2
            
            # Control effort
            cost += R_steering * u_k[0]**2
            cost += R_accel * u_k[1]**2
            
            # Constraints
            opti.subject_to(opti.bounded(self.car.min_steering, u_k[0], self.car.max_steering))
            opti.subject_to(opti.bounded(self.car.min_acceleration, u_k[1], self.car.max_acceleration))
            opti.subject_to(opti.bounded(self.car.min_velocity, x_k[3], self.car.max_velocity))
        
        # Terminal cost
        ref_final = ref_trajectory[-1]
        cost += 5.0 * Q_pos * (X[5, N] - ref_final[0])**2
        cost += 5.0 * Q_pos * (X[6, N] - ref_final[1])**2
        cost += 5.0 * Q_theta * (X[7, N] - ref_final[2])**2
        
        # Set objective
        opti.minimize(cost)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0 if not verbose else 5,
            'print_time': 0,
            'ipopt.max_iter': 100,  # Reduced for speed
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2
        }
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            x_traj = sol.value(X).T
            u_traj = sol.value(U).T
            self.u_last = u_traj[0, :].copy()
            return 0, x_traj, u_traj
            
        except Exception as e:
            if not verbose:
                print(f"MPC solve failed: {e}")
            # Return zero control
            x_traj = np.tile(x0, (N + 1, 1))
            u_traj = np.zeros((N, self.u_dim))
            return -1, x_traj, u_traj

