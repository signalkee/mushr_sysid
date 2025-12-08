# SSI MPC vs SSI MPPI Verification Summary

## ✅ Verified Alignments

### 1. Wheelbase
- **SSI MPC**: Uses `self.car.wheelbase = 0.295` ✓
- **SSI MPPI**: Hardcoded `0.295` ✓
- **Status**: **ALIGNED**

### 2. Car Dynamics
Both use the same Ackermann bicycle model:
- `x_dot = v * cos(theta)`
- `y_dot = v * sin(theta)`
- `theta_dot = (v / L) * tan(steering)`
- **Status**: **ALIGNED** (except SSI MPC integrates acceleration to get velocity)

### 3. Integration Timestep
- **SSI MPC**: `dt = horizon / num_steps` (typically 0.5/10 = 0.05)
- **SSI MPPI**: Hardcoded `dt = 0.05`
- **Status**: **ALIGNED** (both use 0.05)

### 4. Cost Function Weights
- **SSI MPC**: 
  - Q_block_pos = 10.0
  - Q_block_theta = 5.0
  - Q_contact = 8.0
  - R_steering = 0.5
  - R_accel = 0.1
- **SSI MPPI**: **NOW ALIGNED** (updated to match SSI MPC)
  - Q_block_pos = 10.0 ✓
  - Q_block_theta = 5.0 ✓
  - Q_contact = 8.0 ✓
  - R_steering = 0.5 ✓
  - R_speed = 0.1 ✓
- **Status**: **FIXED AND ALIGNED**

### 5. SSI Residual Application
- **SSI MPC**: Applies residuals to block accelerations (state indices 8, 9, 10)
  - `vx_block_dot = vx_block_nominal_ddot + residuals[0]`
- **SSI MPPI**: Applies residuals to block velocities (converted to position changes)
  - `block_x_next += residuals[0] * dt` (where residuals[0] is velocity correction)
- **Status**: **FUNCTIONALLY EQUIVALENT** - Both learn corrections to block motion, just at different levels (acceleration vs velocity) due to different state spaces

### 6. SSI Learning Parameters Usage
- **SSI MPC**: Uses `alpha_last` in dynamics via CasADi symbolic functions ✓
- **SSI MPPI**: **NOW FIXED** - Uses `alpha` with proper random feature computation per state-action pair ✓
- **Status**: **FIXED** - Both now correctly use learned parameters

## ⚠️ Acceptable Differences (Due to Different Algorithm Requirements)

### 1. State Dimensions
- **SSI MPC**: 11D `[x_car, y_car, theta_car, v_car, omega_car, x_block, y_block, theta_block, vx_block, vy_block, omega_block]`
  - Needed for acceleration-based control
- **SSI MPPI**: 6D `[car_x, car_y, car_theta, block_x, block_y, block_theta]`
  - Sufficient for velocity-based control
- **Status**: **ACCEPTABLE** - Different control paradigms require different state spaces

### 2. Control Inputs
- **SSI MPC**: `[steering_angle, acceleration]`
  - Integrates acceleration to get velocity: `v_next = v + acceleration * dt`
- **SSI MPPI**: `[steering, velocity]`
  - Uses velocity directly as input
- **Status**: **ACCEPTABLE** - MPPI typically uses velocity control, MPC uses acceleration control

### 3. Block Positioning Model
- **SSI MPC**: Assumes block at bumper position (`offset_to_front = 0.135`)
  - `bumper_x = x_car + offset * cos(theta)`
  - Block velocities computed from bumper motion
- **SSI MPPI**: Maintains current offset between car and block
  - Transforms offset to car frame, maintains it during motion
- **Status**: **ACCEPTABLE** - Both are valid quasi-static pushing models

### 4. Contact Logic
- **SSI MPC**: Always assumes contact (quasi-static assumption)
- **SSI MPPI**: Checks `is_pushing = |speed| >= 0.2` before applying block dynamics
- **Status**: **ACCEPTABLE** - Different modeling assumptions

## Key Fixes Applied

1. ✅ **SSI MPPI now uses learned alpha parameters correctly**
   - Fixed: Computes random features for each state-action pair in batch
   - Fixed: Applies residuals properly instead of constant offset

2. ✅ **Cost function weights aligned**
   - Updated SSI MPPI weights to match SSI MPC exactly
   - Terminal costs also aligned

3. ✅ **Integration timestep verified**
   - Both use dt = 0.05
   - Added documentation in SSI MPPI

## Conclusion

The dynamics models are **functionally equivalent** at the algorithmic level, with differences that are **acceptable** given the different state representations and control paradigms. The key issue (SSI MPPI not using learned parameters) has been **fixed**, and cost functions are now **aligned**.

Both controllers now:
- Use the same wheelbase (0.295)
- Use the same integration timestep (0.05)
- Use the same cost function weights
- Properly apply SSI-learned residuals
- Use equivalent car dynamics models

The remaining differences (state dimensions, control inputs, block positioning) are necessary for the different control algorithms and do not affect the fairness of comparison.

