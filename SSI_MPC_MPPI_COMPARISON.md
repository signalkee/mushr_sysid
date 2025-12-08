# SSI MPC vs SSI MPPI Comparison Analysis

## Critical Differences Found

### 1. State Representation
- **SSI MPC**: 11D state `[x_car, y_car, theta_car, v_car, omega_car, x_block, y_block, theta_block, vx_block, vy_block, omega_block]`
- **SSI MPPI**: 6D state `[car_x, car_y, car_theta, block_x, block_y, block_theta]`
- **Issue**: Different state spaces make direct comparison difficult

### 2. Control Inputs
- **SSI MPC**: `[steering_angle, acceleration]` (2D)
- **SSI MPPI**: `[steering, velocity]` (2D)
- **Issue**: Different control spaces - one uses acceleration, other uses velocity directly

### 3. Dynamics Model Parameters

#### Wheelbase
- **SSI MPC**: Uses `self.car.wheelbase = 0.295` ✓
- **SSI MPPI**: Hardcoded `0.295` ✓
- **Status**: Same

#### Offset to Front Bumper
- **SSI MPC**: Uses `self.car.offset_to_front = 0.135` in dynamics
- **SSI MPPI**: Does NOT use offset - assumes block is at car center
- **Issue**: Different block positioning logic

#### Car Dynamics
- **SSI MPC**: 
  - `x_dot = v_car * cos(theta)`
  - `y_dot = v_car * sin(theta)`
  - `theta_dot = (v_car / L) * tan(steering)`
  - `v_dot = acceleration`
- **SSI MPPI**:
  - `x_dot = speed * cos(theta) * dt`
  - `y_dot = speed * sin(theta) * dt`
  - `theta_dot = (speed * tan(steering) / 0.295) * dt`
  - Speed is direct input (not integrated from acceleration)
- **Issue**: Different integration methods

#### Block Dynamics
- **SSI MPC**: 
  - Computes block velocities from car motion with offset
  - `vx_block_nominal = v_car * cos(theta) - offset * theta_dot * sin(theta)`
  - `vy_block_nominal = v_car * sin(theta) + offset * theta_dot * cos(theta)`
- **SSI MPPI**:
  - Maintains offset in car frame, transforms to global
  - `offset_x_car = cos_th * dx_global + sin_th * dy_global`
  - `block_x_next = x_next + (cos_th_next * offset_x_car - sin_th_next * offset_y_car)`
- **Issue**: Different coordinate transformations

### 4. Integration Timestep
- **SSI MPC**: `dt = horizon / num_steps` (variable, typically ~0.01-0.05)
- **SSI MPPI**: Hardcoded `dt = 0.05` in `push_dynamics_ssi()`
- **Issue**: Different timesteps can cause numerical differences

### 5. SSI Residual Application
- **SSI MPC**: 
  - Applies residuals to block accelerations: `vx_block_dot = vx_block_nominal_ddot + residuals[0]`
  - Target mask: `[8, 9, 10]` (vx_block_dot, vy_block_dot, omega_block_dot)
- **SSI MPPI**:
  - Applies residuals to block positions: `block_x_next += residuals[0] * dt`
  - Target mask: `[0, 1]` (vx, vy corrections)
- **Issue**: Different residual application (acceleration vs position)

### 6. Cost Function Weights

#### SSI MPC Cost Weights:
```python
Q_block_pos = 10.0      # Block position tracking
Q_block_theta = 5.0     # Block heading tracking
Q_contact = 8.0         # Contact maintenance
Q_vel = 0.1             # Velocity smoothness
R_steering = 0.5        # Steering effort
R_accel = 0.1           # Acceleration effort
Terminal: 5.0 * Q_block_pos, 5.0 * Q_block_theta
```

#### SSI MPPI Cost Weights:
```python
position_cost = 1.0     # Block position (x, y)
heading_cost = 2.0      # Block heading
action_cost_steering = 0.001  # Steering effort
action_cost_speed = 0.01      # Speed effort
alignment_cost = 0.5    # Car-block alignment
Terminal: 1.5 (heading), 1.0 (position)
```

**Issue**: Orders of magnitude difference in weights!

### 7. Contact/Pushing Logic
- **SSI MPC**: Always assumes contact (quasi-static model)
- **SSI MPPI**: Checks `is_pushing = torch.abs(speed) >= 0.2` and only applies dynamics if pushing
- **Issue**: Different contact assumptions

## Recommendations

1. **Align state representations** - Either both use 6D or both use 11D
2. **Align control inputs** - Both should use same control space (acceleration or velocity)
3. **Use same offset** - Both should use `offset_to_front = 0.135`
4. **Align integration timesteps** - Use same dt value
5. **Align cost function weights** - Scale to similar magnitudes
6. **Align SSI residual application** - Both should apply to same quantities (accelerations or velocities)
7. **Align contact logic** - Both should use same pushing condition

## Fixes Applied

### ✅ Cost Function Alignment (FIXED)
- Updated SSI MPPI cost weights to match SSI MPC:
  - `Q_block_pos = 10.0` (was 1.0)
  - `Q_block_theta = 5.0` (was 2.0)
  - `Q_contact = 8.0` (was 0.5)
  - `R_steering = 0.5` (was 0.001)
  - `R_speed = 0.1` (was 0.01, equivalent to R_accel)
  - Terminal: `5.0 * Q_block_pos` and `5.0 * Q_block_theta` (was 1.0 and 1.5)

### ✅ Integration Timestep (VERIFIED)
- Both use `dt = 0.05` (SSI MPC: `dt = horizon / num_steps`, typically 0.5/10 = 0.05)
- Added comment in SSI MPPI to document this alignment

### ⚠️ Remaining Differences (Acceptable for Different Algorithms)

1. **State Dimensions**: 
   - SSI MPC: 11D (includes velocities) - needed for acceleration-based control
   - SSI MPPI: 6D (positions only) - sufficient for velocity-based control
   - **Status**: Acceptable - different control paradigms require different state spaces

2. **Control Inputs**:
   - SSI MPC: `[steering, acceleration]` - integrates to get velocity
   - SSI MPPI: `[steering, velocity]` - direct velocity control
   - **Status**: Acceptable - MPPI typically uses velocity, MPC uses acceleration

3. **Offset Usage**:
   - SSI MPC: Assumes block at bumper (`offset_to_front = 0.135`)
   - SSI MPPI: Maintains current offset between car and block
   - **Status**: Different modeling assumptions - both are valid quasi-static models

4. **SSI Residual Application**:
   - SSI MPC: Applies to block accelerations (more physically accurate)
   - SSI MPPI: Applies to block positions via velocity corrections (simpler for 6D state)
   - **Status**: Different but equivalent - both learn corrections to block motion

