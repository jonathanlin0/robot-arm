tech stack: MuJoCo + RoboSuite
robot arm: SO-101

# Virtual Environment

Run

```
source .venv/bin/activate
```

# Setup

Run the commands all from the root directory

### Download so101 MuJuCo file
taken from https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101

```
./scripts/download_so101_mujoco_model.sh
```

# Tests

```
python -m pytest
```

# Model Information

### so101
- 6 hinge joints
- 6 position actuators
- one-to-one mapping

Joints:
0: shoulder_pan    type=mjJNT_HINGE qpos[0] range=[-1.920, 1.920] rad
1: shoulder_lift   type=mjJNT_HINGE qpos[1] range=[-1.745, 1.745] rad
2: elbow_flex      type=mjJNT_HINGE qpos[2] range=[-1.690, 1.690] rad
3: wrist_flex      type=mjJNT_HINGE qpos[3] range=[-1.658, 1.658] rad
4: wrist_roll      type=mjJNT_HINGE qpos[4] range=[-2.744, 2.744] rad
5: gripper         type=mjJNT_HINGE qpos[5] range=[-0.175, 1.745] rad

Actuators:
0: shoulder_pan    controls=shoulder_pan    ctrl[0] range=[-1.920, 1.920]
1: shoulder_lift   controls=shoulder_lift   ctrl[1] range=[-1.745, 1.745]
2: elbow_flex      controls=elbow_flex      ctrl[2] range=[-1.690, 1.690]
3: wrist_flex      controls=wrist_flex      ctrl[3] range=[-1.658, 1.658]
4: wrist_roll      controls=wrist_roll      ctrl[4] range=[-2.744, 2.841]
5: gripper         controls=gripper         ctrl[5] range=[-0.175, 1.745]

# Methodology

The neural net predicts (dx, dy, dz, dclamp) of the clamp. Then, inverse kinematics is used to calculate how to modify the 5 arms of the so101 to move to the new location: (x + dx, y + dy, z + dz). dclamp isn't part of the IK process. It's just manually controlled.

Inverse kinematics utilizes the Jacobian to find the set of joint changes that most closely gets the gripper to the new position. The $3 \times 5$ Jacobian tells us how the position of the gripper changes wrt each joint change. So, each row corresponds how dx dy or dz changes, and each column corresponds with a different joint. The jacobian gives a local approximation, which is why it is recalculated.

We then find the series of joint changes that minimizes the error of the new position of the gripper from the desired location. This process is done x number of times or until the error is below some threshold.

Entire processes:
- calc gripper location
- calculate xyz error
- calculate jacobian
- choose a small 5-joint correction
- update candidate joint angles
- repeat until close enough or max iterations reached

# Misc

Python version: 3.12
- right now, designed to work on apple silicon chip. untested on normal x86 + nvidia gpu setup