"""
SSI-MPC Demo: Learning Unknown Mass and Friction

This demo shows SSI learning to compensate for WRONG nominal mass/friction parameters.
Uses nonlinear MPC (no linearization).
"""

import gymnasium as gym
from mushr_mujoco_gym.envs.block import MushrBlockEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
from ssi import pose_euler2quat
from ssi.ackermann_model import AckermannCar
from ssi.pushing_ssi_mpc_nonlinear import PushingSSIMpcNonlinear
import time
import torch
import matplotlib.pyplot as plt


def pose_quat2euler(pose):
    return np.array([
        pose[0], 
        pose[1], 
        (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)
    ])


def generate_curved_path(start_pose, dis=1, curvature=0, num_points=100):
    """Generate curved or straight trajectory."""
    x0, y0, theta0 = start_pose
    
    if abs(curvature) < 1e-3:
        # Straight line
        s = (dis / (num_points - 1)) * np.arange(num_points)
        x = x0 + s * np.cos(theta0)
        y = y0 + s * np.sin(theta0)
        traj = np.column_stack([x, y, np.full(num_points, theta0)])
    else:
        # Curved path (arc)
        R = 1.0 / curvature
        delta_theta = dis * curvature
        center_x = x0 - R * np.sin(theta0)
        center_y = y0 + R * np.cos(theta0)
        angle_start = np.arctan2(y0 - center_y, x0 - center_x)
        angles = angle_start + (delta_theta / (num_points - 1)) * np.arange(num_points)
        x = center_x + R * np.cos(angles)
        y = center_y + R * np.sin(angles)
        theta = angles + np.pi / 2
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        traj = np.column_stack([x, y, theta])
    
    return traj


print("="*70)
print("SSI-MPC: Learning Unknown Mass and Friction")
print("="*70)

# ============================================================================
# SETUP: Create mismatch between nominal and true parameters
# ============================================================================

print("\n[1/6] Setting up parameter mismatch...")

# Car model with WRONG nominal parameters
car = AckermannCar()
car.block_mass_nominal = 0.3  # WRONG (will set true to 0.8)
car.block_friction_nominal = 0.2  # WRONG (will set true to 0.5)

TRUE_MASS = 0.8  # kg (actual block mass)
TRUE_FRICTION = 0.5  # (actual friction coefficient)

print(f"   Nominal (wrong):  mass={car.block_mass_nominal} kg, friction={car.block_friction_nominal}")
print(f"   True (unknown):   mass={TRUE_MASS} kg, friction={TRUE_FRICTION}")
print(f"   → SSI must learn to compensate for this mismatch!")

# ============================================================================
# INITIALIZE ENVIRONMENT
# ============================================================================

print("\n[2/6] Creating environment...")
env = gym.make("MushrBlock-v0", render_mode="human", xml_file="sysid_env3.xml")

car_start = [1.0, -1.0, np.pi/2]
block_start = [1.0, -0.7, np.pi/2]

env.reset()
init_state = np.concatenate((pose_euler2quat(car_start), pose_euler2quat(block_start)))
obs = env.unwrapped.set_init_states(init_state)

# ============================================================================
# CREATE SSI-MPC CONTROLLER (Nonlinear)
# ============================================================================

print("\n[3/6] Initializing nonlinear SSI-MPC...")

# Random features setup
n_rf = 20
n_inputs = 11  # all states as features
kernel_std = 0.3
lr = 0.2  # Higher learning rate for faster adaptation

omega = np.random.normal(0.0, kernel_std, (n_rf, n_inputs))
b = np.random.uniform(0.0, 2.0 * np.pi, (n_rf, 1))

rf_dict = {
    'n_rf': n_rf,
    'omega': omega,
    'b': b,
    'input': list(range(11)),  # all states
    'target': [8, 9, 10],  # learn block accelerations
    'lr': lr
}

# Create nonlinear MPC solver
mpc_solver = PushingSSIMpcNonlinear(
    name='mushr_nonlinear',
    car=car,
    horizon=0.15,
    num_steps=8,  # Small for speed
    rf_dict=rf_dict,
    true_mass=TRUE_MASS,
    true_friction=TRUE_FRICTION
)

# ============================================================================
# GENERATE TRAJECTORY
# ============================================================================

print("\n[4/6] Generating trajectory...")
# Use FULL CIRCLE trajectory (same as main.py)
trajectory = generate_curved_path(block_start, dis=np.pi, curvature=0.5, num_points=50)
print(f"   Full circle path: {len(trajectory)} waypoints")

# ============================================================================
# CONTROL LOOP
# ============================================================================

print("\n[5/6] Running control loop...")
print("-"*70)

# Settle
for i in range(50):
    obs, _, _, _, _ = env.step([0, 0])

# Control
max_steps = 500
errors = []
alpha_norms = []
solve_times = []
car_positions = []      # Track car trajectory
block_positions = []    # Track block trajectory
step = 0
ref_idx = 0

start_time = time.time()

try:
    while step < max_steps:
        # Extract state
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        # Build 11D state
        state = np.array([
            car_euler[0], car_euler[1], car_euler[2],  # car x, y, theta
            0.15,  # v_car (assume constant pushing velocity)
            0.0,   # omega_car
            block_euler[0], block_euler[1], block_euler[2],  # block x, y, theta
            0.15 * np.cos(block_euler[2]),  # vx_block (approximate)
            0.15 * np.sin(block_euler[2]),  # vy_block
            0.0    # omega_block
        ])
        
        # Find reference index
        block_pos = block_euler[:2]
        dist_to_waypoints = np.linalg.norm(trajectory[:, :2] - block_pos, axis=1)
        ref_idx = dist_to_waypoints.argmin()
        ref_idx = min(ref_idx + 2, len(trajectory) - 1)  # Look ahead
        
        # Generate reference for MPC
        ref_traj = np.zeros((mpc_solver.num_steps + 1, 3))
        for i in range(mpc_solver.num_steps + 1):
            idx = min(ref_idx + i, len(trajectory) - 1)
            ref_traj[i, :] = trajectory[idx, :]
        
        # Track error
        error = np.linalg.norm(block_pos - trajectory[ref_idx, :2])
        errors.append(error)
        
        # Track learning
        alpha_norm = np.linalg.norm(mpc_solver.alpha_last)
        alpha_norms.append(alpha_norm)
        
        # Store trajectory history
        car_positions.append([car_euler[0], car_euler[1]])
        block_positions.append([block_euler[0], block_euler[1]])
        
        # Solve MPC
        t0 = time.time()
        status, x_traj, u_traj = mpc_solver.solve_mpc(state, ref_traj, dt=0.02, verbose=False)
        solve_time = time.time() - t0
        solve_times.append(solve_time)
        
        if status != 0:
            print(f"   [WARNING] MPC failed at step {step}")
            action = np.array([0.0, 0.15])  # Default: straight at min push velocity
        else:
            # Extract control
            steering = u_traj[0, 0]
            # Use planned velocity from next state
            planned_vel = x_traj[1, 3] if x_traj.shape[0] > 1 else 0.15
            velocity = max(0.15, planned_vel)  # Ensure minimum push velocity
            action = np.array([steering, velocity])
        
        # Apply control
        obs, _, _, _, _ = env.step(action)
        
        # Progress
        if step % 50 == 0:
            print(f"   Step {step:3d} | Ref {ref_idx:2d}/{len(trajectory)} | "
                  f"Error: {error:.3f} m | ||α||: {alpha_norm:.3f} | "
                  f"MPC: {solve_time*1000:.0f} ms")
        
        # Check goal
        if ref_idx >= len(trajectory) - 1 and error < 0.2:
            print(f"\n   Goal reached at step {step}!")
            break
        
        step += 1
        
except KeyboardInterrupt:
    print("\n   Interrupted")

elapsed = time.time() - start_time
env.close()

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("Results")
print("="*70)
print(f"Steps: {step}, Time: {elapsed:.1f} s, Freq: {step/elapsed:.1f} Hz")
print(f"Mean error: {np.mean(errors):.3f} m, Final error: {errors[-1]:.3f} m")
print(f"Mean MPC time: {np.mean(solve_times)*1000:.0f} ms")
print(f"Final ||α||: {alpha_norms[-1]:.3f} (learning magnitude)")
print(f"Final ref: {ref_idx}/{len(trajectory)}")
print("="*70)

# ============================================================================
# PLOTS
# ============================================================================

print("\n[6/6] Creating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Tracking error
ax = axes[0, 0]
ax.plot(errors, 'b-', linewidth=2)
ax.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='10cm')
ax.set_xlabel('Step')
ax.set_ylabel('Tracking Error (m)')
ax.set_title('Tracking Error Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# SSI learning (alpha norm)
ax = axes[0, 1]
ax.plot(alpha_norms, 'g-', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('||α|| (Learning Parameters)')
ax.set_title('SSI Learning Progress')
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, f'Mass error: {(TRUE_MASS - car.block_mass_nominal)/TRUE_MASS*100:.0f}%\n' +
                   f'Friction error: {(TRUE_FRICTION - car.block_friction_nominal)/TRUE_FRICTION*100:.0f}%',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# MPC solve time
ax = axes[1, 0]
ax.plot([t*1000 for t in solve_times], 'r-', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Time (ms)')
ax.set_title('MPC Solve Time (Nonlinear)')
ax.grid(True, alpha=0.3)

# Trajectory comparison
ax = axes[1, 1]
# Convert to arrays
car_positions = np.array(car_positions)
block_positions = np.array(block_positions)

# Plot reference
ax.plot(trajectory[:, 0], trajectory[:, 1], 'b--', linewidth=3, label='Reference', alpha=0.7)

# Plot actual trajectories
ax.plot(block_positions[:, 0], block_positions[:, 1], 'r-', linewidth=2, label='Block (actual)')
ax.plot(car_positions[:, 0], car_positions[:, 1], 'g-', linewidth=1.5, label='Car (actual)', alpha=0.7)

# Start and end points
ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start', zorder=5)
ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=12, label='Goal', zorder=5)
ax.plot(block_positions[-1, 0], block_positions[-1, 1], 'rx', markersize=12, label='Final pos', zorder=5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Trajectory Tracking (Reference vs Actual)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('ssi_learning_results.png', dpi=150, bbox_inches='tight')
print("   Saved: ssi_learning_results.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: Did SSI Learn the Mass/Friction Mismatch?")
print("="*70)
print(f"Parameter mismatch:")
print(f"  - Mass:     {car.block_mass_nominal} kg (nominal) vs {TRUE_MASS} kg (true)")
print(f"  - Friction: {car.block_friction_nominal} (nominal) vs {TRUE_FRICTION} (true)")
print(f"\nSSI learning indicator:")
print(f"  - Initial ||α||: {alpha_norms[0]:.3f}")
print(f"  - Final ||α||:   {alpha_norms[-1]:.3f}")
print(f"  - Change: {(alpha_norms[-1] - alpha_norms[0]):.3f}")
print(f"\nIf ||α|| increases significantly, SSI is learning to compensate!")
print("="*70)

