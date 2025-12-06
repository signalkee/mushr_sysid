"""
Three-Way Comparison Experiment: SSI-MPC vs MPPI vs SSI-MPPI

Compares three controllers on the same pushing task:
1. SSI-MPC: MPC with online mass/friction learning
2. MPPI: Sampling-based MPC (no learning)
3. SSI-MPPI: MPPI with online mass/friction learning

Shows the effect of adding SSI to different base controllers.
"""

import gymnasium as gym
from mushr_mujoco_gym.envs.block import MushrBlockEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
from ssi import pose_euler2quat, PushingSSIMPPI
from ssi.ackermann_model import AckermannCar
from ssi.pushing_ssi_mpc_nonlinear import PushingSSIMpcNonlinear
from path_tracking_controller import PushingController
import time
import torch
import matplotlib.pyplot as plt


def pose_quat2euler(pose):
    return np.array([
        pose[0], pose[1], 
        (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)
    ])


def generate_curved_path(start_pose, dis=1, curvature=0, num_points=100):
    """Generate trajectory."""
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


def run_controller(env, controller, controller_name, trajectory, max_steps=700):
    """Generic function to run any controller and collect metrics."""
    print(f"\n{'='*70}")
    print(f"Running: {controller_name}")
    print(f"{'='*70}")
    
    errors = []
    solve_times = []
    block_positions = []
    car_positions = []
    alpha_norms = [] if hasattr(controller, 'alpha') else None
    
    # Settle
    obs, _, _, _, _ = env.step([0, 0])
    for i in range(50):
        obs, _, _, _, _ = env.step([0, 0])
    
    print("   Running control...")
    start_time = time.time()
    
    for step in range(max_steps):
        # Extract state
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])
        block_euler[2] = block_euler[2] - np.pi/2
        
        obs_euler = np.concatenate((car_euler, block_euler))
        block_pos = block_euler[:2]
        
        # Get reference
        if hasattr(controller, 'get_reference_index'):
            ref_idx = controller.get_reference_index(obs_euler)
        else:
            # For SSI-MPC controller
            dist_to_waypoints = np.linalg.norm(trajectory[:, :2] - block_pos, axis=1)
            ref_idx = min(dist_to_waypoints.argmin() + 2, len(trajectory) - 1)
        
        # Track error
        error = np.linalg.norm(block_pos - trajectory[ref_idx, :2])
        errors.append(error)
        
        # Track positions
        car_positions.append([car_euler[0], car_euler[1]])
        block_positions.append([block_euler[0], block_euler[1]])
        
        # Track learning (if applicable)
        if alpha_norms is not None:
            if hasattr(controller, 'alpha_last'):
                # SSI-MPC nonlinear
                alpha_norms.append(np.linalg.norm(controller.alpha_last))
            elif hasattr(controller, 'alpha') and isinstance(controller.alpha, np.ndarray):
                # SSI-MPPI
                alpha_norms.append(np.linalg.norm(controller.alpha))
            elif hasattr(controller, 'mpc_solver') and hasattr(controller.mpc_solver, 'alpha_last'):
                # SSI-MPC gym wrapper
                alpha_norms.append(np.linalg.norm(controller.mpc_solver.alpha_last))
        
        # Compute control
        t0 = time.time()
        
        if controller_name == "SSI-MPC":
            # SSI-MPC specific command
            state = np.array([
                car_euler[0], car_euler[1], car_euler[2], 0.15, 0.0,
                block_euler[0], block_euler[1], block_euler[2],
                0.15 * np.cos(block_euler[2]), 0.15 * np.sin(block_euler[2]), 0.0
            ])
            ref_traj = np.zeros((controller.num_steps + 1, 3))
            for i in range(controller.num_steps + 1):
                idx = min(ref_idx + i, len(trajectory) - 1)
                ref_traj[i, :] = trajectory[idx, :]
            
            status, x_traj, u_traj = controller.solve_mpc(state, ref_traj, dt=0.02, verbose=False)
            if status == 0:
                steering = u_traj[0, 0]
                velocity = max(0.15, x_traj[1, 3] if x_traj.shape[0] > 1 else 0.15)
                action = np.array([steering, velocity])
            else:
                action = np.array([0.0, 0.15])
        elif controller_name == "SSI-MPPI":
            # SSI-MPPI has command method
            action = controller.command(obs)
        else:
            # MPPI (original) uses ctrl.command
            action = controller.ctrl.command(obs)
        
        solve_times.append(time.time() - t0)
        
        # Step
        obs, _, _, _, _ = env.step(action)
        
        # Progress
        if step % 100 == 0:
            alpha_str = f", ||α||: {alpha_norms[-1]:.2f}" if alpha_norms else ""
            print(f"   Step {step:3d} | Ref {ref_idx:2d}/{len(trajectory)} | "
                  f"Error: {error:.3f} m{alpha_str}")
        
        # Check goal
        if ref_idx >= len(trajectory) - 1 and error < 0.2:
            print(f"   Goal reached at step {step}!")
            break
    
    elapsed = time.time() - start_time
    
    results = {
        'name': controller_name,
        'errors': np.array(errors),
        'alpha_norms': np.array(alpha_norms) if alpha_norms else None,
        'solve_times': np.array(solve_times),
        'block_positions': np.array(block_positions),
        'car_positions': np.array(car_positions),
        'total_time': elapsed,
        'total_steps': len(errors),
        'final_ref': ref_idx
    }
    
    alpha_str = f", ||α||={alpha_norms[-1]:.2f}" if alpha_norms else ""
    print(f"   Done: {len(errors)} steps, {elapsed:.1f}s, error={np.mean(errors):.3f}m{alpha_str}")
    
    return results


def plot_three_way_comparison(results_list, trajectory):
    """Create comprehensive three-way comparison plot."""
    print("\nCreating three-way comparison plots...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = ['blue', 'red', 'green']
    labels = [r['name'] for r in results_list]
    
    # 1. Trajectory comparison (LARGE)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=3, 
             label='Reference', alpha=0.7, zorder=1)
    
    for i, results in enumerate(results_list):
        ax1.plot(results['block_positions'][:, 0], 
                results['block_positions'][:, 1], 
                color=colors[i], linewidth=2, label=f"{results['name']} Block", alpha=0.7)
    
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start', zorder=5)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=15, label='Goal', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Trajectory Tracking Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Tracking error over time
    ax2 = fig.add_subplot(gs[0, 2])
    for i, results in enumerate(results_list):
        ax2.plot(results['errors'], color=colors[i], linewidth=2, 
                label=results['name'], alpha=0.7)
    ax2.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='10cm')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Tracking Error')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. SSI Learning comparison
    ax3 = fig.add_subplot(gs[1, 2])
    for i, results in enumerate(results_list):
        if results['alpha_norms'] is not None:
            ax3.plot(results['alpha_norms'], color=colors[i], linewidth=2, 
                    label=f"{results['name']}", alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('||α|| (Learning)')
    ax3.set_title('SSI Learning Progress')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    if not any(r['alpha_norms'] is not None for r in results_list):
        ax3.text(0.5, 0.5, 'No learning\n(MPPI only)', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    
    # 4. Computation time
    ax4 = fig.add_subplot(gs[2, 0])
    for i, results in enumerate(results_list):
        ax4.plot(results['solve_times']*1000, color=colors[i], linewidth=1.5, 
                label=results['name'], alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Computation Time per Step')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Error box plot
    ax5 = fig.add_subplot(gs[2, 1])
    error_data = [r['errors'] for r in results_list]
    bp = ax5.boxplot(error_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax5.set_ylabel('Error (m)')
    ax5.set_title('Error Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Performance metrics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = [['Metric'] + labels, ['─'*15] + ['─'*10]*len(results_list)]
    
    metrics = [
        ('Mean Error (m)', lambda r: f"{np.mean(r['errors']):.3f}"),
        ('Std Error (m)', lambda r: f"{np.std(r['errors']):.3f}"),
        ('Final Error (m)', lambda r: f"{r['errors'][-1]:.3f}"),
        ('Mean Time (ms)', lambda r: f"{np.mean(r['solve_times'])*1000:.0f}"),
        ('Steps', lambda r: f"{r['total_steps']}"),
        ('Progress', lambda r: f"{r['final_ref']}/{len(trajectory)}"),
        ('Learning', lambda r: f"{r['alpha_norms'][-1]:.1f}" if r['alpha_norms'] is not None else "N/A"),
    ]
    
    for metric_name, metric_func in metrics:
        row = [metric_name] + [metric_func(r) for r in results_list]
        stats_text.append(row)
    
    table = ax6.table(cellText=stats_text, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    for i in range(len(labels) + 1):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Performance Metrics', fontsize=11, fontweight='bold', pad=15)
    
    plt.suptitle('Three-Way Controller Comparison: SSI-MPC vs MPPI vs SSI-MPPI', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('experiment_three_way.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_three_way.png")
    plt.close()


def main():
    print("="*70)
    print("THREE-WAY EXPERIMENT: SSI-MPC vs MPPI vs SSI-MPPI")
    print("="*70)
    print("\nComparing three controllers:")
    print("  1. SSI-MPC:  MPC with SSI learning (nonlinear)")
    print("  2. MPPI:     Sampling-based (no learning)")
    print("  3. SSI-MPPI: MPPI with SSI learning")
    print("\nAll use WRONG nominal mass/friction. SSI controllers must learn!")
    
    # Common setup
    car_start = [1.0, -1.0, np.pi/2]
    block_start = [1.0, -0.7, np.pi/2]
    
    # Trajectory (semicircle)
    trajectory = generate_curved_path(block_start, dis=np.pi, curvature=0.5, num_points=100)
    print(f"\nTrajectory: Semicircle, {len(trajectory)} waypoints")
    
    max_steps = 700
    results_list = []
    
    # ========================================================================
    # EXPERIMENT 1: SSI-MPC
    # ========================================================================
    
    env1 = gym.make("MushrBlock-v0", render_mode="human", xml_file="sysid_env3.xml")
    env1.reset()
    init_state = np.concatenate((pose_euler2quat(car_start), pose_euler2quat(block_start)))
    env1.unwrapped.set_init_states(init_state)
    
    # Create SSI-MPC
    car_ssi_mpc = AckermannCar()
    car_ssi_mpc.block_mass_nominal = 0.3
    car_ssi_mpc.block_friction_nominal = 0.2
    
    omega = np.random.normal(0.0, 0.3, (20, 11))
    b = np.random.uniform(0.0, 2.0 * np.pi, (20, 1))
    rf_dict = {'n_rf': 20, 'omega': omega, 'b': b, 
               'input': list(range(11)), 'target': [8, 9, 10], 'lr': 0.02}
    
    ssi_mpc_controller = PushingSSIMpcNonlinear(
        'ssi_mpc', car_ssi_mpc, 0.15, 8, rf_dict, true_mass=0.8, true_friction=0.5
    )
    
    ssi_mpc_results = run_controller(env1, ssi_mpc_controller, "SSI-MPC", trajectory, max_steps)
    env1.close()
    results_list.append(ssi_mpc_results)
    
    # ========================================================================
    # EXPERIMENT 2: MPPI (Original)
    # ========================================================================
    
    env2 = gym.make("MushrBlock-v0", render_mode="rgb_array", xml_file="sysid_env3.xml")
    env2.reset()
    env2.unwrapped.set_init_states(init_state)
    
    mppi_controller = PushingController()
    mppi_controller.set_trajectory(torch.tensor(trajectory, dtype=torch.float32))
    
    mppi_results = run_controller(env2, mppi_controller, "MPPI", trajectory, max_steps)
    env2.close()
    results_list.append(mppi_results)
    
    # ========================================================================
    # EXPERIMENT 3: SSI-MPPI
    # ========================================================================
    
    env3 = gym.make("MushrBlock-v0", render_mode="rgb_array", xml_file="sysid_env3.xml")
    env3.reset()
    env3.unwrapped.set_init_states(init_state)
    
    ssi_mppi_controller = PushingSSIMPPI(
        device='cpu',
        n_rf=20,
        lr=0.2,
        kernel_std=0.3,
        nominal_mass=0.3,
        nominal_friction=0.2,
        true_mass=0.8,
        true_friction=0.5
    )
    ssi_mppi_controller.set_trajectory(torch.tensor(trajectory, dtype=torch.float32))
    
    ssi_mppi_results = run_controller(env3, ssi_mppi_controller, "SSI-MPPI", trajectory, max_steps)
    env3.close()
    # results_list.append(ssi_mppi_results)
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for results in results_list:
        print(f"\n{results['name']}:")
        print(f"  Mean error: {np.mean(results['errors']):.3f} m")
        print(f"  Std error:  {np.std(results['errors']):.3f} m")
        print(f"  Mean time:  {np.mean(results['solve_times'])*1000:.0f} ms")
        print(f"  Progress:   {results['final_ref']}/{len(trajectory)}")
        if results['alpha_norms'] is not None:
            print(f"  Learning:   ||α|| = {results['alpha_norms'][-1]:.2f}")
    
    # Find best accuracy
    errors_mean = [np.mean(r['errors']) for r in results_list]
    best_idx = np.argmin(errors_mean)
    
    print(f"\n{'='*70}")
    print(f"BEST ACCURACY: {results_list[best_idx]['name']}")
    print(f"  Mean error: {errors_mean[best_idx]:.3f} m")
    
    # Improvements
    for i, results in enumerate(results_list):
        if i != best_idx:
            improvement = (errors_mean[i] - errors_mean[best_idx]) / errors_mean[i] * 100
            print(f"  {improvement:.1f}% better than {results['name']}")
    print(f"{'='*70}")
    
    # Plot
    plot_three_way_comparison(results_list, trajectory)
    
    print("\n[SUCCESS] Experiment complete! Check experiment_three_way.png")
    
    # Return results for analysis
    return results_list


if __name__ == "__main__":
    results = main()

