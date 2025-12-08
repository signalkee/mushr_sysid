# Pushing Model Contact Analysis

## Executive Summary

**Critical Issues Found:**
1. ❌ **Loss of contact is NOT properly accounted for** - the spring force always applies regardless of distance
2. ❌ **Block can be moved even when not in contact** - no distance-based contact condition
3. ⚠️ **Contact model has fundamental physics issues** - virtual spring acts at any distance

---

## 1. Is the pushing model accounting for loss of contact?

### Answer: **NO, loss of contact is NOT properly modeled**

### Current Implementation Issues:

#### A. Nonlinear Model (`pushing_ssi_mpc_nonlinear.py`)
**Lines 154-157:**
```python
# Virtual spring force (pushes block to bumper position)
k_spring = 50.0  # Spring stiffness
F_contact_x = -k_spring * error_x
F_contact_y = -k_spring * error_y
```

**Problem:** The spring force is **always active** regardless of the distance between car and block. The force scales linearly with distance but never becomes zero, meaning:
- If block is 1 meter away, it still experiences a 50N force pulling it toward the car
- This is physically unrealistic - contact forces should only exist when objects are touching

#### B. Linear Model (`pushing_ssi_mpc.py`)
**Lines 331-335:**
```python
# Block velocities follow car (quasi-static, fully linearized)
# Assume block moves with nominal car velocity
vx_block_next = v_nom * cos_theta_ref
vy_block_next = v_nom * sin_theta_ref
omega_block_next = (v_nom / L) * steering_k
```

**Problem:** The block **always follows the car** with no contact checking. This is a quasi-static assumption that doesn't model contact loss at all.

#### C. MPPI Model (`pushing_cost_function.py`)
**Lines 212, 244-261:**
```python
is_pushing = torch.abs(speed) >= min_push_velocity
# ... block dynamics only apply if is_pushing is True
block_x_next = torch.where(is_pushing, block_x_push, block_x)
```

**Better:** This model does check if the car is moving fast enough, but it doesn't check if the block is actually in contact with the car.

---

## 2. If contact is lost, should the car be able to update the block pose?

### Answer: **NO - if contact is lost, the block should NOT be influenced by the car**

**Correct Physics:**
- Contact force should only exist when: `distance(block, car_bumper) < contact_threshold`
- When contact is lost, the block should:
  - Continue with its current velocity (subject to friction)
  - NOT be affected by car movements
  - Only resume contact if the car catches up and re-establishes contact

**Current Implementation:**
- ❌ Spring force always applies (nonlinear model)
- ❌ Block always follows car (linear model)
- ⚠️ Only velocity threshold checked (MPPI model)

---

## 3. Is Q_contact the only term that accounts for keeping in contact?

### Answer: **NO - there are TWO different contact mechanisms:**

#### A. `Q_contact` - Cost Function Term (Optimization Incentive)
**Location:** MPC cost functions in both linear and nonlinear models

**Purpose:** Penalizes deviation between block position and car bumper in the **optimization objective**

**Files:**
- `pushing_ssi_mpc.py` line 264, 367-368: `Q_contact = 8.0`
- `pushing_ssi_mpc_nonlinear.py` line 244, 275-276: `Q_contact = 10.0`

**Mechanism:** 
- Adds cost: `Q_contact * (block_x - bumper_x)² + Q_contact * (block_y - bumper_y)²`
- **Encourages** the MPC optimizer to keep block near bumper
- This is a **soft constraint** - contact can still be violated if other objectives conflict

#### B. `F_contact` - Dynamics Force (Physics Simulation)
**Location:** Only in nonlinear model (`pushing_ssi_mpc_nonlinear.py`)

**Purpose:** Actually **applies forces** to the block based on distance from bumper

**Lines 154-166:**
```python
# Virtual spring force (pushes block to bumper position)
k_spring = 50.0
F_contact_x = -k_spring * error_x
F_contact_y = -k_spring * error_y
# Applied in dynamics: F = ma
vx_block_ddot = (F_contact_x + F_friction_x) / m_block
```

**Mechanism:**
- This is a **hard physics constraint** - the force directly affects block acceleration
- Acts as a virtual spring pulling block toward bumper
- Always active regardless of distance (PROBLEM!)

**Summary:**
- `Q_contact`: Soft incentive in cost function (MPC optimization)
- `F_contact`: Hard force in dynamics (physics simulation)
- **Both should respect contact distance thresholds, but currently neither does**

---

## 4. What is the role of the spring contact in the pushing model?

### Answer: **The spring contact model is a SIMPLIFICATION that is PHYSICALLY INCORRECT**

#### Intended Role:
The spring contact model (lines 154-157 in `pushing_ssi_mpc_nonlinear.py`) is meant to simulate:
- Contact forces between car bumper and block
- Restoring force when block deviates from ideal contact position
- A simple way to model the complex physics of pushing

#### Current Implementation:
```python
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
```

#### Problems:

1. **No Distance Threshold:**
   - Force applies at ANY distance (even 10 meters away!)
   - Real contact forces require physical contact (distance ≈ 0)

2. **Always Attractive:**
   - Force always pulls block toward bumper
   - No repulsion if block is "inside" the car
   - No distinction between contact and no-contact states

3. **No Contact Condition:**
   - Missing: `if distance(block, bumper) < contact_threshold: apply_force()`
   - Missing: `else: force = 0` (block moves independently)

#### What It Should Be:

A **conditional spring-damper model**:
```python
contact_distance = sqrt(error_x² + error_y²)
contact_threshold = 0.15  # meters (block size + small margin)

if contact_distance < contact_threshold:
    # Apply spring-damper force when in contact
    F_contact_x = -k_spring * error_x - c_damping * vx_block
    F_contact_y = -k_spring * error_y - c_damping * vy_block
else:
    # No contact - no force from car
    F_contact_x = 0.0
    F_contact_y = 0.0
```

---

## 5. Is the pushing model correct? Verification

### Answer: **NO - The model has significant physics issues**

### Model Verification Checklist:

#### ✅ Correct Aspects:
1. **Car dynamics (Ackermann model)** - correctly implemented
2. **Friction model** - viscous friction opposing motion
3. **State representation** - proper 11D state space
4. **SSI learning framework** - correctly augments dynamics with residuals

#### ❌ Incorrect Aspects:

##### 1. **Contact Force Always Active** (CRITICAL)
- **Location:** `pushing_ssi_mpc_nonlinear.py:154-157`
- **Issue:** Spring force applies at any distance
- **Fix Needed:** Add distance threshold check

##### 2. **No Contact Loss Handling** (CRITICAL)
- **Location:** All models
- **Issue:** Block always influenced by car, even when far away
- **Fix Needed:** Conditional force application based on contact distance

##### 3. **Unrealistic Spring Constant**
- **Location:** `pushing_ssi_mpc_nonlinear.py:155`
- **Issue:** `k_spring = 50.0` is very high - creates unrealistic forces
- **Fix Needed:** Use realistic contact stiffness or add damping

##### 4. **Missing Contact Geometry**
- **Issue:** No check for actual collision/contact geometry
- **Fix Needed:** Verify block is actually touching car bumper (not just close)

##### 5. **Quasi-Static Assumption in Linear Model**
- **Location:** `pushing_ssi_mpc.py:331-335`
- **Issue:** Block always follows car exactly (no dynamics)
- **Note:** This is an intentional simplification for convex optimization, but not physically accurate

---

## Recommended Fixes

### Priority 1: Add Contact Distance Check

```python
def _physics_based_dynamics(self):
    # ... existing car dynamics ...
    
    # Contact detection
    bumper_x = self.x_car + offset * cs.cos(self.theta_car)
    bumper_y = self.y_car + offset * cs.sin(self.theta_car)
    
    error_x = self.x_block - bumper_x
    error_y = self.y_block - bumper_y
    contact_distance = cs.sqrt(error_x**2 + error_y**2)
    
    # Contact threshold (block size + margin)
    contact_threshold = 0.15  # meters
    
    # Conditional contact force
    k_spring = 50.0
    # Smooth approximation: force decays to zero beyond threshold
    # Use sigmoid or smooth step function
    contact_ratio = cs.fmax(0, 1.0 - contact_distance / contact_threshold)
    contact_ratio = cs.fmin(1.0, contact_ratio)  # Clamp to [0, 1]
    
    F_contact_x = -k_spring * error_x * contact_ratio
    F_contact_y = -k_spring * error_y * contact_ratio
    
    # ... rest of dynamics ...
```

### Priority 2: Update Q_contact to Match

The cost function should also respect contact loss:
```python
# Only penalize contact deviation if in contact
contact_distance = sqrt((block_x - bumper_x)² + (block_y - bumper_y)²)
if contact_distance < contact_threshold:
    cost += Q_contact * (block_x - bumper_x)²
    cost += Q_contact * (block_y - bumper_y)²
# Else: no contact cost (block can move freely)
```

### Priority 3: Add Contact State to Cost

Consider adding a term that explicitly encourages maintaining contact:
```python
# Large penalty if contact is lost
contact_penalty = 100.0 * cs.fmax(0, contact_distance - contact_threshold)**2
cost += contact_penalty
```

---

## Summary

| Question | Answer | Status |
|----------|--------|--------|
| Is loss of contact accounted for? | **NO** | ❌ Critical Issue |
| Should block move when contact lost? | **NO** | ❌ Not enforced |
| Is Q_contact the only contact term? | **NO** - also F_contact | ⚠️ Two mechanisms |
| What is spring contact role? | Simplification (incorrect) | ❌ Needs fix |
| Is model correct? | **NO** - physics issues | ❌ Needs revision |

**Main Problem:** The spring force model assumes contact is always maintained, which is physically unrealistic. Contact should be conditional based on distance, and forces should only apply when objects are actually touching.

