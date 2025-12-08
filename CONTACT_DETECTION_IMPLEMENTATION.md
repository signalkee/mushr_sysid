# Contact Detection Implementation

## Overview

Added contact detection to both SSI MPC and SSI MPPI to ensure that block dynamics only apply when the car is actually in contact with the block. This prevents unrealistic behavior where the block moves even when the car is far away.

## Implementation Details

### Contact Conditions

Contact is considered **active** when **both** conditions are met:

1. **Distance Condition**: Block is within contact threshold of car bumper
   - `contact_distance < contact_threshold` (0.15 meters)
   - Accounts for block size (~0.1m) plus a small margin

2. **Velocity Condition**: Car is moving forward fast enough to push
   - `v_car >= min_push_velocity` (0.1 m/s)
   - Prevents pushing when car is stationary or moving too slowly

### Contact Parameters

```python
contact_threshold = 0.15  # meters - maximum distance for contact
min_push_velocity = 0.1    # m/s - minimum car velocity to push
friction_coeff = 0.3      # Friction coefficient when not in contact
```

### Contact Detection Logic

**SSI MPC (CasADi symbolic):**
```python
# Distance from block to bumper
dx_contact = x_block - bumper_x
dy_contact = y_block - bumper_y
contact_distance = sqrt(dx_contact² + dy_contact² + ε)

# Smooth step functions for differentiability
is_in_contact = smooth_step(contact_threshold - contact_distance)
is_pushing = smooth_step(v_car - min_push_velocity)
contact_active = is_in_contact * is_pushing
```

**SSI MPPI (NumPy/PyTorch):**
```python
# Distance from block to bumper
dx_contact = x_block - bumper_x
dy_contact = y_block - bumper_y
contact_distance = sqrt(dx_contact² + dy_contact² + ε)

# Binary contact check
is_in_contact = 1.0 if contact_distance < contact_threshold else 0.0
is_pushing = 1.0 if v_car >= min_push_velocity else 0.0
contact_active = is_in_contact * is_pushing
```

### Block Dynamics When In Contact

When `contact_active = 1.0`:
- Block follows car bumper (quasi-static pushing model)
- SSI residuals are applied to block accelerations
- Block velocities and accelerations are computed from car motion

### Block Dynamics When NOT In Contact

When `contact_active = 0.0`:
- Block accelerations from car are **zero**
- Block velocities decay due to friction:
  ```
  vx_block_dot = -friction_coeff * vx_block
  vy_block_dot = -friction_coeff * vy_block
  omega_block_dot = -friction_coeff * omega_block
  ```
- Block continues moving with its current velocity (subject to friction)
- Car cannot influence block position

### Sliding and Contact Loss

**Sliding Detection:**
- If block slides away from bumper (distance > threshold), contact is lost
- Block immediately stops receiving forces from car
- Block velocities decay due to friction

**Contact Re-establishment:**
- If car catches up and gets within threshold AND is moving fast enough, contact resumes
- Block dynamics switch back to pushing mode

## Physics Correctness

### ✅ What This Fixes

1. **No Force at Distance**: Block no longer experiences forces when car is far away
2. **Realistic Contact**: Only applies pushing forces when actually touching
3. **Sliding Behavior**: Block can slide away and lose contact
4. **Friction Decay**: Block velocities decay when not in contact (realistic)

### Key Behaviors

1. **Contact Active**:
   - Block follows car bumper
   - SSI residuals apply
   - Block can be pushed

2. **Contact Lost**:
   - Block moves independently
   - Velocities decay due to friction
   - Car cannot push block

3. **Contact Re-established**:
   - When car gets close enough AND is moving
   - Pushing resumes immediately

## Implementation Locations

### SSI MPC
- **File**: `ssi/pushing_ssi_mpc.py`
- **Function**: `_augmented_dynamics()` (lines 130-190)
- **Method**: Uses CasADi smooth step functions for differentiability

### SSI MPPI
- **File**: `ssi/pushing_ssi_mppi.py`
- **Functions**: 
  - `predict_next_state_numpy()` (lines 261-320) - NumPy version
  - `push_dynamics_ssi()` (lines 348-410) - PyTorch version
- **Method**: Uses binary checks (0/1) for efficiency in sampling

## Testing Recommendations

1. **Contact Loss Test**: Move car away from block - block should stop moving
2. **Sliding Test**: Push block sideways - should lose contact if distance > threshold
3. **Re-contact Test**: Move car back to block - should resume pushing
4. **Low Velocity Test**: Car moving < 0.1 m/s should not push block
5. **Distance Threshold Test**: Block at exactly 0.15m should lose contact

## Parameters Tuning

- **`contact_threshold`**: Increase if block is losing contact too easily
- **`min_push_velocity`**: Increase if car is pushing when it shouldn't
- **`friction_coeff`**: Increase for faster velocity decay when not in contact

## Summary

✅ **Contact detection is now implemented in both SSI MPC and SSI MPPI**

The models now correctly:
- Only apply block dynamics when in contact
- Account for sliding and contact loss
- Apply friction when not in contact
- Prevent unrealistic force application at distance

This makes the pushing model physically realistic and prevents the block from being influenced by the car when they are not in contact.

