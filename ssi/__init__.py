"""
SSI-MPC: Simultaneous System Identification and Model Predictive Control

Implementations for:
- Crazyflie quadrotor control with wind disturbance learning
- MuSHR car pushing control with mass/friction learning
"""

from .ackermann_model import AckermannCar, BlockParams
from .pushing_ssi_mpc import PushingSSIMpc
from .pushing_ssi_mpc_nonlinear import PushingSSIMpcNonlinear
from .pushing_ssi_mppi import PushingSSIMPPI
from .pushing_ssi_mpc_gym import PushingSSIMPCController, pose_quat2euler, pose_euler2quat

__all__ = [
    'AckermannCar',
    'BlockParams',
    'PushingSSIMpc',
    'PushingSSIMpcNonlinear',
    'PushingSSIMPPI',
    'PushingSSIMPCController',
    'pose_quat2euler',
    'pose_euler2quat',
]

