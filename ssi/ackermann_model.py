"""
Ackermann steering car model parameters for MuSHR platform.
"""
import numpy as np


class AckermannCar:
    """
    Parameters for the MuSHR 1/10 scale rally car.
    """
    def __init__(self):
        # Vehicle geometry
        self.wheelbase = 0.295  # meters (distance between front and rear axles)
        self.offset_to_front = 0.135  # meters (offset from car center to front bumper)
        
        # Vehicle mass and inertia (approximate for MuSHR)
        self.mass = 1.5  # kg (approximate)
        self.inertia = 0.05  # kg*m^2 (approximate rotational inertia)
        
        # Block properties (nominal values - SSI will learn deviations)
        self.block_mass_nominal = 0.5  # kg (nominal mass)
        self.block_friction_nominal = 0.3  # dimensionless (nominal friction coefficient)
        self.block_inertia_nominal = 0.01  # kg*m^2 (approximate)
        
        # Physical constants
        self.gravity = 9.81  # m/s^2
        
        # Control constraints
        self.max_steering = 0.17  # rad (from existing controller)
        self.min_steering = -0.17  # rad
        self.max_velocity = 0.21  # m/s
        self.min_velocity = 0.0  # m/s (forward only)
        self.max_acceleration = 2.0  # m/s^2 (reasonable default)
        self.min_acceleration = -2.0  # m/s^2
        
        # State constraints
        self.max_omega = 2.0  # rad/s (max angular velocity)
        self.workspace_limit = 5.0  # meters (workspace bounds)


class BlockParams:
    """
    Block physical properties.
    """
    def __init__(self, mass=0.5, friction=0.3, size=0.1):
        self.mass = mass  # kg
        self.friction = friction  # coefficient of friction
        self.size = size  # meters (characteristic dimension)
        self.inertia = mass * size**2 / 6.0  # approximate as uniform cube

