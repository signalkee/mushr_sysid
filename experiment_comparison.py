"""
Experiment: SSI-MPC vs MPPI Comparison

Compares the performance of:
1. SSI-MPC (with online mass/friction learning)
2. MPPI (sampling-based, no learning)

Both controllers push a block along the same trajectory.
"""

import gymnasium as gym
from mushr_mujoco_gym.envs.block import MushrBlockEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
from ssi import pose_euler2quat
from ssi.ackermann_model import AckermannCar
from ssi.pushing_ssi_mpc_nonlinear import PushingSSIMpcNonlinear
from path_tracking_controller import PushingController
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
        s = (dis / (num_points - 1)) * np.arange(num_points)
        x = x0 + s * np.cos(theta0)
        y = y0 + s * np.sin(theta0)
        traj = np.column_stack([x, y, np.full(num_points, theta0)])
    else:
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


def run_ssi_mpc(env, trajectory, max_steps=400):
    """Run SSI-MPC controller and collect metrics."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: SSI-MPC (with online learning)")
    print("="*70)
    
    # Setup with WRONG parameters (SSI must learn)
    car = AckermannCar()
    car.block_mass_nominal = 0.3  # WRONG
    car.block_friction_nominal = 0.2  # WRONG
    TRUE_MASS = 0.8
    TRUE_FRICTION = 0.5
    
    print(f"   Nominal: mass={car.block_mass_nominal} kg, friction={car.block_friction_nominal}")
    print(f"   True:    mass={TRUE_MASS} kg, friction={TRUE_FRICTION}")
    
    # Random features
    n_rf = 20
    omega = np.random.normal(0.0, 0.3, (n_rf, 11))
    b = np.random.uniform(0.0, 2.0 * np.pi, (n_rf, 1))
    
    rf_dict = {
        'n_rf': n_rf,
        'omega': omega,
        'b': b,
        'input': list(range(11)),
        'target': [8, 9, 10],
        'lr': 0.01
    }
    
    # Create controller
    mpc_solver = PushingSSIMpcNonlinear(
        name='mushr_ssi',
        car=car,
        horizon=0.1,
        num_steps=8,
        rf_dict=rf_dict,
        true_mass=TRUE_MASS,
        true_friction=TRUE_FRICTION
    )
    
    # Metrics
    errors = []
    alpha_norms = []
    solve_times = []
    block_positions = []
    car_positions = []
    
    # Settle
    for i in range(50):
        obs, _, _, _, _ = env.step([0, 0])
    
    print("   Running control loop...")
    start_time = time.time()
    
    for step in range(max_steps):
        # Extract state
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        # State vector
        state = np.array([
            car_euler[0], car_euler[1], car_euler[2],
            0.15, 0.0,
            block_euler[0], block_euler[1], block_euler[2],
            0.15 * np.cos(block_euler[2]), 0.15 * np.sin(block_euler[2]), 0.0
        ])
        
        # Reference
        block_pos = block_euler[:2]
        dist_to_waypoints = np.linalg.norm(trajectory[:, :2] - block_pos, axis=1)
        ref_idx = min(dist_to_waypoints.argmin() + 2, len(trajectory) - 1)
        
        ref_traj = np.zeros((mpc_solver.num_steps + 1, 3))
        for i in range(mpc_solver.num_steps + 1):
            idx = min(ref_idx + i, len(trajectory) - 1)
            ref_traj[i, :] = trajectory[idx, :]
        
        # Error
        error = np.linalg.norm(block_pos - trajectory[ref_idx, :2])
        errors.append(error)
        
        # Learning
        alpha_norm = np.linalg.norm(mpc_solver.alpha_last)
        alpha_norms.append(alpha_norm)
        
        # History
        car_positions.append([car_euler[0], car_euler[1]])
        block_positions.append([block_euler[0], block_euler[1]])
        
        # Solve
        t0 = time.time()
        status, x_traj, u_traj = mpc_solver.solve_mpc(state, ref_traj, dt=0.02, verbose=False)
        solve_times.append(time.time() - t0)
        
        if status != 0:
            action = np.array([0.0, 0.15])
        else:
            steering = u_traj[0, 0]
            velocity = max(0.15, x_traj[1, 3] if x_traj.shape[0] > 1 else 0.15)
            action = np.array([steering, velocity])
        
        obs, _, _, _, _ = env.step(action)
        
        if step % 100 == 0:
            print(f"   Step {step:3d} | Ref {ref_idx:2d}/{len(trajectory)} | "
                  f"Error: {error:.3f} m | ||α||: {alpha_norm:.2f}")
        
        if ref_idx >= len(trajectory) - 1 and error < 0.2:
            print(f"   Goal reached at step {step}!")
            break
    
    elapsed = time.time() - start_time
    
    results = {
        'name': 'SSI-MPC',
        'errors': np.array(errors),
        'alpha_norms': np.array(alpha_norms),
        'solve_times': np.array(solve_times),
        'block_positions': np.array(block_positions),
        'car_positions': np.array(car_positions),
        'total_time': elapsed,
        'total_steps': len(errors),
        'final_ref': ref_idx
    }
    
    print(f"\n   Results: {len(errors)} steps, {elapsed:.1f}s, "
          f"error={np.mean(errors):.3f}m, ||α||={alpha_norms[-1]:.2f}")
    
    return results


def run_mppi(env, trajectory, max_steps=400):
    """Run MPPI controller and collect metrics."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: MPPI (sampling-based, no learning)")
    print("="*70)
    
    # Create MPPI controller
    push_controller = PushingController()
    push_controller.set_trajectory(torch.tensor(trajectory, dtype=torch.float32))
    
    # Metrics
    errors = []
    solve_times = []
    block_positions = []
    car_positions = []
    
    # Settle
    for i in range(50):
        obs, _, _, _, _ = env.step([0, 0])
    
    print("   Running control loop...")
    start_time = time.time()
    
    for step in range(max_steps):
        # Extract state
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        obs_euler = np.concatenate((car_euler, block_euler))
        
        # Reference
        ref_idx = push_controller.get_reference_index(obs_euler)
        
        # Error
        block_pos = block_euler[:2]
        error = np.linalg.norm(block_pos - trajectory[ref_idx, :2])
        errors.append(error)
        
        # History
        car_positions.append([car_euler[0], car_euler[1]])
        block_positions.append([block_euler[0], block_euler[1]])
        
        # Solve
        t0 = time.time()
        action = push_controller.ctrl.command(obs)
        solve_times.append(time.time() - t0)
        
        obs, _, _, _, _ = env.step(action)
        
        if step % 100 == 0:
            print(f"   Step {step:3d} | Ref {ref_idx:2d}/{len(trajectory)} | "
                  f"Error: {error:.3f} m")
        
        if ref_idx >= len(trajectory) - 1 and error < 0.2:
            print(f"   Goal reached at step {step}!")
            break
    
    elapsed = time.time() - start_time
    
    results = {
        'name': 'MPPI',
        'errors': np.array(errors),
        'alpha_norms': None,  # MPPI doesn't learn
        'solve_times': np.array(solve_times),
        'block_positions': np.array(block_positions),
        'car_positions': np.array(car_positions),
        'total_time': elapsed,
        'total_steps': len(errors),
        'final_ref': ref_idx
    }
    
    print(f"\n   Results: {len(errors)} steps, {elapsed:.1f}s, "
          f"error={np.mean(errors):.3f}m")
    
    return results


def plot_comparison(ssi_results, mppi_results, trajectory):
    """Create comprehensive comparison plots."""
    print("\nCreating comparison plots...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Trajectory comparison (large plot)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=3, 
             label='Reference', alpha=0.7, zorder=1)
    ax1.plot(ssi_results['block_positions'][:, 0], ssi_results['block_positions'][:, 1], 
             'b-', linewidth=2, label='SSI-MPC Block', alpha=0.8)
    ax1.plot(mppi_results['block_positions'][:, 0], mppi_results['block_positions'][:, 1], 
             'r-', linewidth=2, label='MPPI Block', alpha=0.8)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, 
             label='Start', zorder=5)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=15, 
             label='Goal', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Trajectory Tracking Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Tracking error over time
    ax2 = fig.add_subplot(gs[0, 2])
    min_len = min(len(ssi_results['errors']), len(mppi_results['errors']))
    ax2.plot(ssi_results['errors'][:min_len], 'b-', linewidth=2, label='SSI-MPC', alpha=0.7)
    ax2.plot(mppi_results['errors'][:min_len], 'r-', linewidth=2, label='MPPI', alpha=0.7)
    ax2.axhline(0.1, color='g', linestyle='--', alpha=0.5, label='10cm')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Tracking Error')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. SSI Learning (only for SSI-MPC)
    ax3 = fig.add_subplot(gs[1, 2])
    if ssi_results['alpha_norms'] is not None:
        ax3.plot(ssi_results['alpha_norms'], 'g-', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('||α||')
        ax3.set_title('SSI Learning (SSI-MPC only)')
        ax3.grid(True, alpha=0.3)
        ax3.text(0.5, 0.95, f"Final: {ssi_results['alpha_norms'][-1]:.1f}",
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 4. Computation time comparison
    ax4 = fig.add_subplot(gs[2, 0])
    min_len = min(len(ssi_results['solve_times']), len(mppi_results['solve_times']))
    ax4.plot(ssi_results['solve_times'][:min_len]*1000, 'b-', linewidth=1.5, 
             label='SSI-MPC', alpha=0.7)
    ax4.plot(mppi_results['solve_times'][:min_len]*1000, 'r-', linewidth=1.5, 
             label='MPPI', alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Computation Time per Step')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Error histogram
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(ssi_results['errors'], bins=30, alpha=0.6, label='SSI-MPC', color='blue', edgecolor='black')
    ax5.hist(mppi_results['errors'], bins=30, alpha=0.6, label='MPPI', color='red', edgecolor='black')
    ax5.axvline(np.mean(ssi_results['errors']), color='b', linestyle='--', linewidth=2)
    ax5.axvline(np.mean(mppi_results['errors']), color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Error (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = [
        ['Metric', 'SSI-MPC', 'MPPI'],
        ['─'*20, '─'*12, '─'*12],
        ['Mean Error (m)', f"{np.mean(ssi_results['errors']):.3f}", 
         f"{np.mean(mppi_results['errors']):.3f}"],
        ['Final Error (m)', f"{ssi_results['errors'][-1]:.3f}", 
         f"{mppi_results['errors'][-1]:.3f}"],
        ['Max Error (m)', f"{np.max(ssi_results['errors']):.3f}", 
         f"{np.max(mppi_results['errors']):.3f}"],
        ['Mean Time (ms)', f"{np.mean(ssi_results['solve_times'])*1000:.1f}", 
         f"{np.mean(mppi_results['solve_times'])*1000:.1f}"],
        ['Total Steps', f"{ssi_results['total_steps']}", 
         f"{mppi_results['total_steps']}"],
        ['Total Time (s)', f"{ssi_results['total_time']:.1f}", 
         f"{mppi_results['total_time']:.1f}"],
        ['Progress', f"{ssi_results['final_ref']}/{len(trajectory)}", 
         f"{mppi_results['final_ref']}/{len(trajectory)}"],
    ]
    
    table = ax6.table(cellText=stats_text, cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig('experiment_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_comparison.png")
    plt.close()


def main():
    print("="*70)
    print("EXPERIMENT: SSI-MPC vs MPPI Comparison")
    print("="*70)
    print("\nObjective: Compare two controllers on the same pushing task:")
    print("  1. SSI-MPC: Online learning of unknown mass/friction")
    print("  2. MPPI: Sampling-based MPC (no learning)")
    
    # Common setup
    car_start = [1.0, -1.0, np.pi/2]
    block_start = [1.0, -0.7, np.pi/2]
    
    # Generate trajectory (semicircle - same as main.py line 132)
    r = 10.0
    trajectory = generate_curved_path(block_start, dis=np.pi*r*1/2, curvature=1/r, num_points=50)
    print(f"\nTrajectory: Semicircle with {len(trajectory)} waypoints")
    
    # Experiment 1: SSI-MPC
    env1 = gym.make("MushrBlock-v0", render_mode="human", xml_file="sysid_env3.xml")
    env1.reset()
    init_state = np.concatenate((pose_euler2quat(car_start), pose_euler2quat(block_start)))
    obs1 = env1.unwrapped.set_init_states(init_state)
    
    ssi_results = run_ssi_mpc(env1, trajectory, max_steps=1900)
    env1.close()
    
    # Experiment 2: MPPI
    env2 = gym.make("MushrBlock-v0", render_mode="rgb_array", xml_file="sysid_env3.xml")
    env2.reset()
    obs2 = env2.unwrapped.set_init_states(init_state)
    
    mppi_results = run_mppi(env2, trajectory, max_steps=1000)
    env2.close()
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nSSI-MPC:")
    print(f"  Mean error: {np.mean(ssi_results['errors']):.3f} m")
    print(f"  Mean solve time: {np.mean(ssi_results['solve_times'])*1000:.1f} ms")
    print(f"  Learning: ||α|| grew from 0 to {ssi_results['alpha_norms'][-1]:.2f}")
    print(f"  Progress: {ssi_results['final_ref']}/{len(trajectory)}")
    
    print(f"\nMPPI:")
    print(f"  Mean error: {np.mean(mppi_results['errors']):.3f} m")
    print(f"  Mean solve time: {np.mean(mppi_results['solve_times'])*1000:.1f} ms")
    print(f"  Learning: N/A (no online learning)")
    print(f"  Progress: {mppi_results['final_ref']}/{len(trajectory)}")
    
    # Determine winner
    ssi_error = np.mean(ssi_results['errors'])
    mppi_error = np.mean(mppi_results['errors'])
    
    print(f"\n{'='*70}")
    if ssi_error < mppi_error:
        improvement = (mppi_error - ssi_error) / mppi_error * 100
        print(f"WINNER: SSI-MPC ({improvement:.1f}% better tracking accuracy)")
    elif mppi_error < ssi_error:
        improvement = (ssi_error - mppi_error) / ssi_error * 100
        print(f"WINNER: MPPI ({improvement:.1f}% better tracking accuracy)")
    else:
        print("RESULT: Tie!")
    print(f"{'='*70}")
    
    # Plot
    plot_comparison(ssi_results, mppi_results, trajectory)
    
    print("\n[SUCCESS] Experiment complete! Check experiment_comparison.png")


if __name__ == "__main__":
    main()

