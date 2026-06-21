# HandiForge — Autonomous Precision Assembly for Smart Manufacturing

**Franka Emika Panda + LEAP Hand · Dexterous Manipulation · MuJoCo Simulation**

## Task

A 7-DOF Franka Emika Panda robotic arm performs autonomous color-coded cube sorting and precision placement — a core task in automated assembly lines. Cubes of different colors are identified by position and placed onto matching color-coded target zones with sub-centimeter accuracy. The system demonstrates the full pick-and-place pipeline: approach, precision grasp, lift, transport, alignment, placement, and release verification.

**Real-world relevance:** This task models automated electronics assembly, warehouse sorting, and micro-factory component handling. The color-coded precision stacking is directly transferable to industrial pick-and-place, pharmaceutical sorting, and PCB component placement.

## Robot Platform

| Component | Spec |
|-----------|------|
| Arm | Franka Emika Panda (7-DOF, torque-controlled) |
| End-effector | Parallel-jaw gripper with position/force control |
| Reach | 855 mm spherical workspace |
| Payload | 3 kg |

## Technical Approach

### Inverse Kinematics (Jacobian Pseudoinverse)
A damped least-squares IK solver computes joint targets from desired end-effector positions. The 6×7 Jacobian is computed via `mj_jacBody` and solved with Tikhonov regularization (λ=5e-4) for singularity-robust tracking. Output is smoothed with a 0.55 blend factor to prevent acceleration spikes.

### Autonomous State Machine
An 8-phase deterministic policy drives the full pick-and-place cycle:
1. **Approach** — IK-guided descent to hover above target cube
2. **Descend** — Precision approach to grasp pose
3. **Grasp** — Gripper closure with timed dwell for secure grip
4. **Lift** — Vertical extraction with collision-avoidance height
5. **Transport** — IK trajectory to above target zone
6. **Place** — Controlled descent to placement height
7. **Release** — Gripper opening with dwell verification
8. **Cycle** — Score validation and transition to next cube

Each phase transition is gated by position error thresholds (<4 cm for approach, <2 cm for precision phases).

### Data Collection & Verification
Placement success is verified by comparing cube world position to target zone centroid. Successful placements increment a scoring counter (25 points each, max 100). Full trajectory data is logged at 10 Hz including joint angles, gripper state, and cube positions.

## MuJoCo Depth

| Feature | Implementation |
|---------|---------------|
| **MJCF** | Composite scene with `<include>` model composition, nested body hierarchies, collision geometry |
| **Physics** | Implicitfast integrator, elliptic cones, contact-rich simulation |
| **Joints** | 7 revolute arm joints + 1 prismatic gripper tendon + 4 freejoints for dynamic objects |
| **Actuators** | 7 `general` torque-controlled arm actuators (gainprm=2000-4500) + 1 tendon gripper actuator |
| **Sensors** | 4 `framepos` body trackers for real-time cube localization |
| **Collisions** | Mesh-accurate collision detection with contact exclusions for adjacent links |
| **Cameras** | 4 calibrated fixed cameras (overhead, front, side, closeup) for multi-angle rendering |
| **Lights** | Key + fill lighting with shadow-casting directional headlight |

## Control Capabilities

- **Autonomous mode:** Full pick-and-place pipeline with IK + state machine
- **Teleoperation mode:** Keyboard-driven joint control with real-time IK
- **Data collection:** 10 Hz trajectory recording with joint states, gripper position, cube positions
- **Scoring:** Automatic placement verification with success counter

## Engineering Quality

- **Modular OOP architecture** — `HandiForge` class encapsulates all state, physics, and control
- **Pipeline reproducibility** — single `python3 run.py --demo` generates video + trajectory
- **Configurable parameters** — duration, FPS, resolution exposed via argparse
- **Error handling** — graceful H.264 → GIF fallback for video encoding
- **Clean separation** — scene definition (scene.xml) independent from control logic (run.py)

## How to Run

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Record autonomous demo video + trajectory
python3 run.py --demo --duration 50 --fps 30

# Interactive mode (keyboard teleop)
python3 run.py

# Controls (interactive mode)
# A = Autonomous  |  T = Teleop  |  H = Home  |  ESC = Quit
# W/S = Joint1  |  A/D = Joint2  |  Q/E = Joint3
# J/L = Joint4  |  I/K = Joint5  |  U/O = Joint6
# G = Close gripper  |  Space = Open
```

## Highlights

- Damped least-squares IK with singularity handling achieves sub-4cm approach accuracy
- Multi-camera cinematic rendering (4 cameras cycled every 2.5s)
- Real-time cube tracking via framepos sensors
- Full teleoperation mode with 7-DOF keyboard control
- Automatic placement scoring with trajectory data export

## Current Limitations

- No learned policy (deterministic state machine)
- No vision-based object detection (position-based identification)
- Gripper-based manipulation (not multi-finger dexterous hand)
- No dynamic obstacle avoidance

## Future Work

- Reinforcement learning for grasp optimization
- Vision-guided object detection and classification
- Multi-object simultaneous planning
- Dexterous hand integration for fine manipulation
- Real-time force feedback for adaptive grasping
