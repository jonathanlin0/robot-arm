Objective: make robot arm stack orange box on top of the blue box.

How: control the robot arm (via actuators) from some policy. train in sim and then try to zero shot real. ideally combine ML and RL.

Tasks (in order)
- [ ] start with perfect observation - i.e., xyz coordinates of boxes. policy takes in coordinates of boxes, joint angles, box velocities, and joint velocities of arm and outputs gripper coordinate deltas. IK used to calculate the individual joint actions
- [ ] replace perfect xyz coordinates of boxes with camera input. a CV model predicts the squares xyz coordinates and the predictions are used as input to the action model. image data automatically generated via mujoco: taking screenshots of camera view and grabbing labels directly from internal state. essentially unlimited data can be generated. will likely need some sequential model to also predict cube velocity.
- [ ] remove separate CV training pipeline. the CV model produces some latent representation of the camera (i.e., visual encoder). another MLP takes in joint positions and produces latent vector. these values are concactonated as input to policy that produces gripper coordinates
- [ ] modify model to output direct actuator action delta instead of gripper delta. This gets rid of IK reliance. we want to eventually get rid of IK for this specific project since actuators may not be perfect, so nontrivial chance that the joint actions derived from IK that are assumed in a perfect world wouldn't be accurate.

Why: i want to learn more about robotics, specifically from an ML and RL perspective. this is my first experience with robotics.


tech stack: MuJoCo + RoboSuite
robot arm: SO-101

# Virtual Environment

Run

```
source .venv/bin/activate
python -m pip install -r requirements.txt
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

The Cartesian controller uses an orientation-aware version of IK. In
addition to the 3 x 5 position Jacobian, it uses the 3 x 5 rotational
Jacobian to bias the gripper's approach axis toward world-down. The SO-101
model's gripper approach direction is the `gripperframe` site's local +X
axis.

Position remains the primary objective. The orientation correction is
projected into the position Jacobian's null space, which means it uses joint
motion that should not change the gripper's XYZ position to first order.
Roll around the approach axis remains unconstrained. Position and tool-axis
convergence are reported separately because a top-down orientation is not
physically reachable everywhere in the arm's workspace.

# Misc

Python version: 3.12
- right now, designed to work on apple silicon chip. untested on normal x86 + nvidia gpu setup


# AI GENERATED STUFF. NOT HUMAN REVIEWED

# Stacking demonstrations and policy training

The current success condition is a released orange-on-blue stack that remains
stable for **0.5 seconds**, after orange has been grasped and lifted. Holding
orange alone no longer ends an episode.

Generate full demonstrations from randomized positions around the robot's home:

The home gripper position is `(0.40, 0.00, 0.25)` metres, configured by
`DEFAULT_START_POSITION` near the top of `src/environment.py`. The generator's
`START_POSITION_HALF_RANGE = (0.04, 0.04, 0.02)` samples each coordinate
independently and uniformly at reset: X 36–44 cm, Y −4–4 cm, Z 23–27 cm.
Set this range to `(0, 0, 0)` to generate fixed-start demonstrations.

The shared reset solves IK and initializes the open claw directly at the sampled
pose, without advancing simulation time or adding policy history. Start positions
and cube placements use the environment's existing RNG. Failed starts are
discarded through the generator's normal
retry limit. Saved settings record the start distribution, and the initial
observation/accepted-target tensors record the actual starting pose.

Supervised pretraining uses the saved varied starts, while live validation and
demo playback keep the fixed home position. Waypoint-start mode still overrides
the home pose when enabled. Fixed-start and randomized-start demonstration files
can be combined when their scene, action, and physics settings match.

```bash
.venv/bin/python scripts/generate_pickup_demonstrations.py
.venv/bin/python scripts/generate_pickup_demonstrations.py --examples 10
```

The generator records approach, descent, grasp, lift back to the orange
waypoint, horizontal transport above blue, lowering, release, and retreat as
needed. Every recorded command uses the same Gym action adapter as the policy.
Transport and placement track the cube's actual offset from the gripper.
Only successful released stacks are saved; failed attempts get a fresh UUID
and seed. Each invocation adds `NUMBER_OF_EXAMPLES` episodes (80% train, 20% test),
with a 600-action episode limit. Existing files are preserved.

`scripts/pretrain_pickup.py` now reads these full stacking demonstrations and
evaluates from home using the recorded stacking tolerances. Its episode limit
defaults to 600 actions. Old pickup-only demonstrations are rejected, even if
their tensor dimensions match. The filenames and 49-observation/56-token
layout are unchanged.

At each pretraining validation check, the terminal reports full stacking
success and `orange grasp=count/episodes (percentage)`. An episode counts as
grasped once both jaws have held orange clear of the table, even if the cube
is subsequently dropped or released. The grasp count and rate are also
included in validation history and W&B metrics.

Pretraining predicts actions at every real timestep of each sequence in one
forward pass. `PretrainingConfig.history_length` defaults to 384 (19.2 seconds
at 20 Hz); `sequence_stride=256` splits longer episodes into `[0:384]`,
`[256:640]`, etc., stopping when the final action is covered. Each label is
trained once per epoch: overlap remains visible as context but is excluded
from the later chunk's loss. Each chunk uses local positional indices, so
later chunks provide less history near their supervised beginning than a
full rolling inference window does.

All sequence tensors are prepared before the DataLoader is created, with
right padding, a padding attention mask, and a separate loss mask. Causal
attention prevents future actions from leaking into predictions. Batches
trim their all-padding trailing columns before the forward pass. `batch_size`
now counts episodes/chunks (default 8; sweep candidates 4, 8, 16); losses and
metrics average over supervised action timesteps, not padding or chunk count.
Policy inference still chooses only the latest real output and retains at
most 384 tokens. Checkpoints retain their own configured history length.

Run pretraining checks with `.venv/bin/python scripts/pretrain_pickup.py --test`
or `.venv/bin/python -m pytest test/`. The dedicated pretraining tests are in
`test/test_pretrain_*.py`; `--smoke` still runs a small training session and
real simulator evaluation on the saved data.

Run a fresh training session from the repository root:

```bash
python src/train.py
```

PPO's configurable reset behavior is independent of demonstration generation:
its current defaults still start at the normal orange waypoint. Set
`start_at_orange_waypoint=False` to also train the approach from home. Control
preparation in `PPOTrainingConfig` in `src/train.py`:

```python
start_at_orange_waypoint: bool = True
recovery_start_probability: float = 0.0
recovery_xy_offset_range: tuple[float, float] = (0.03, 0.05)
recovery_height_offset_range: tuple[float, float] = (0.03, 0.05)
recovery_closed_gripper_probability: float = 0.5
```

Set `start_at_orange_waypoint=False` to restore ordinary starts for new runs.
Set `recovery_start_probability=0` to keep every prepared start at the open
waypoint. When preparation is enabled, the helper in `src/waypoint_start.py`
first reaches the waypoint on every reset, including vector worker auto-resets.
Recovery starts then move beside the orange cube: the radial horizontal offset
is sampled uniformly between 3 and 5 cm with a random direction, and height
between 3 and 5 cm above the cube's current center. Half of recovery starts
have a closed gripper; the other half are open. These choices use the seeded
environment RNG, and preparation checks that the cube remains on the table
without an acquired grasp. The policy chooses how to recover from the prepared
state; no retry controller takes over during the episode.

Preparation checks the actual gripper pose and accepted Cartesian target for
five consecutive control steps and raises an error if a movement cannot settle
within 200 preparation steps. It does not consume any of the
400 policy steps, produce training rewards, or enter the transformer's history.
With preparation enabled, the waypoint flag starts true so descent rewards
are immediately active;
`orange_waypoint_reach_rate` will therefore be 1 once episodes complete.
Stack success still requires release and the continuous 0.5-second stability check.

`PPOTrainingConfig.start_at_orange_waypoint` defaults to `True` and is used
directly by training without an override. Checkpoints save all preparation
settings. W&B records them with the `environment_` prefix, including
`environment_start_at_orange_waypoint` and the effective
`environment_recovery_start_probability` (zero when preparation is disabled).
The sweep keeps the default preparation mix; these settings are not additional
sweep parameters. Existing rendering and diagnostic scripts continue using the
checkpoint's waypoint-start flag, with recovery sampling disabled for comparable
evaluations rather than sampling the training recovery mix.
Training `success_rate` combines both start types; it is not a waypoint-only
score. Reset and step info include `episode_start_type` for inspecting the mix.

`PPOTrainingConfig` in `src/train.py` controls the temporal policy:

| Setting | Default | Meaning |
| --- | --- | --- |
| `history_length` | 384 | Sliding window in policy timesteps (19.2 seconds at 20 Hz) |
| `transformer_embedding_dim` | 128 | Learned token embedding width |
| `transformer_layers` | 3 | Causal transformer layers |
| `transformer_heads` | 4 | Attention heads; must divide the embedding width |
| `transformer_feedforward_dim` | 512 | Feedforward width inside each transformer layer |
| `model_dim` | 128 | Hidden width of each actor/value MLP head |
| `model_layers` | 2 | Hidden layers in each actor/value MLP head |
| `sde_xyz_log_std_init` | `math.log(0.2 / 7.1)` | Initial gSDE noise-weight log SD for each XYZ action column |
| `sde_gripper_log_std_init` | `math.log(0.5 / 7.1)` | Initial gSDE noise-weight log SD for the gripper action column |

The separate gSDE settings initialize trainable noise parameters, with no fixed
floor or ceiling. At the reference actor feature norm of 7.1, these
defaults give effective action SDs of approximately 0.2 for XYZ and 0.5 for the
gripper, before action clipping. Effective SD still depends on the observation
and learned actor features. The 7.1 reference was measured with tanh MLP hidden
layers; ReLU can change the effective SD. Check the logged action SDs when
changing the architecture or activation. `log_std_init` and
`minimum_gripper_standard_deviation` apply when gSDE is disabled. Loading a
checkpoint preserves its learned noise parameters; new initialization settings
take effect when starting a fresh policy.

Each token contains `[observation[t], action[t-1], accepted_target_xyz[t-1]]`:
49 observation values, four action components, and three commanded
Cartesian target coordinates in world meters, for **56 values** total. The
waypoint-reached flag is retained in episode metrics but excluded from model
inputs. History records the issued action after clipping to the action bounds. The XYZ features
come from the controller's committed target after that action, including
workspace clipping and any successful IK backtracking. If all IK attempts fail,
the previous accepted target is retained. This is the intended command that
survived controller processing; the measured gripper position can still lag it.

The Gym environment passes a copied `target_gripper_position` in reset/step
info, and the history wrapper appends it alongside the matching observation and
action. Episode-start and padding masks are separate from the 56 token values.
At reset the action is zero, the target is the controller's initialized target
(the final accepted IK target when starting at the waypoint, otherwise the
measured reset gripper position), and the start marker is set. Valid tokens
appear in chronological order, with unused slots padded on the right. Each
parallel environment owns and resets its own window.

The shared transformer uses learned position/start embeddings, zero dropout,
and causal attention. Its latest valid token feeds separate actor and value
MLP heads with ReLU hidden activations. Transformer feedforward layers use
GELU, and the action mean still uses a final tanh before gSDE noise is added.
All these networks train together with PPO. PPO stores complete
history snapshots in its dictionary rollout buffer, so shuffled minibatches
and time-limit bootstrapping use the same context as action collection.

Rollout metrics now include `action_std_x/y/z/gripper.txt` and
`action_clip_fraction.txt`. Standard deviations summarize the actual action
distribution before clipping; they can change with the learned representation
even when the configured gSDE noise parameters stay fixed.

`orange_waypoint_reach_rate` tracks the fraction of completed episodes whose
orange pregrasp waypoint flag was reached, including episodes that later fail
or time out. It uses the same rolling episode window as the episode reward
metrics (100 episodes by default), and is logged once per rollout to W&B and
`data/orange_waypoint_reach_rate.txt`. Sweep runs write the file under
`data/wandb/<run_id>/` instead. Values use the same 0–1 scale as `success_rate`
(`0.75` means 75%); the value is `nan` until an episode has finished.

Every 100 training rollouts, training pauses to measure `no_var_success_rate`
over 100 complete episodes with `deterministic=True`. Episodes run sequentially
in one separate environment in the training process, with one PyTorch CPU
thread during evaluation. No evaluation workers are created. The training
thread setting is restored afterward, and evaluation does not add training
timesteps or transitions to PPO's buffer.

`PPOTrainingConfig.evaluation_interval_rollouts`, `evaluation_episodes`, and
`evaluation_seed` default to `100`, `100`, and `20000`. Every evaluation uses
the same seeds 20000–20099, the same history/reward settings and episode limit,
and the configured waypoint-start flag, with recovery sampling disabled.
Evaluations run at the end of rollout collection, before that rollout's PPO
update, matching the existing metric timing.

W&B records `no_var_success_rate` on the 0–1 scale at rollout steps 100, 200,
300, etc. It is included in the same log entry as the ordinary training metrics.
`data/no_var_success_rate.txt` (or `data/wandb/<run_id>/no_var_success_rate.txt`)
has one row per training rollout: measured rates at evaluation rollouts and
`nan` on other rows, preserving alignment with the existing text files. The
ordinary training `success_rate` and sweep objective remain unchanged.

Checkpoints save the transformer settings, history shape, and training config.
`scripts/demo.py` restores the same history wrapper, reward
settings, episode limit, and waypoint-start mode. Checkpoints without the
waypoint-start setting use ordinary starts. Start a new training run for this token layout;
older flat-observation and 54- or 57-value history checkpoints cannot be used by
the updated playback. Demonstrations with the old 50-value observation layout
must also be regenerated for pretraining; existing files are not converted.
Training still saves to `checkpoints/ppo_cube_stacker.zip` and writes metrics
to `data/` by default; set the corresponding config paths for separate runs.

To inspect the training demonstrations themselves, without loading a model:

```bash
.venv/bin/mjpython scripts/demo.py --pretrain-data --workers 3
```

This randomly selects up to twelve distinct completed episodes from `data/train/`
and renders their saved robot/cube poses in a grid with 3 rows and 4 columns
directly into `stack_demo.mp4`. It
includes every recorded state, holds shorter episodes on their final frame,
and leaves unused grid cells black. Terminal output identifies each panel's
episode UUID. The green terminal border reflects the saved demonstration's
success status; playback does not rerun physics or evaluate a policy.
`--pretrain-data` and `--repeat-wandb` are mutually exclusive.

W&B records the new settings as `model_history_length`,
`model_transformer_embedding_dim`, `model_transformer_layers`,
`model_transformer_heads`, `model_transformer_feedforward_dim`,
`model_sde_xyz_log_std_init`, and `model_sde_gripper_log_std_init`.
They can be added to `WANDB_SWEEP_CONFIG["parameters"]`, and
`--repeat-wandb RUN_ID` restores them when present. The supported `model_dim`
and `model_layers` parameters control the MLP heads.

Run the current focused Bayesian sweep with `python src/train.py --wandb`.
To add a parallel agent to that same sweep, run
`python src/train.py --wandb SWEEP_ID` in another terminal. A short ID uses
`WANDB_ENTITY_NAME` and `WANDB_PROJECT_NAME`; a full
`entity/project/sweep_id` path is also accepted. Each invocation runs one
agent, which requests trials from the shared sweep until stopped. Joining
uses the existing sweep's search configuration.

New sweeps search eight parameters:

| Sweep parameter | Range | Sampling |
| --- | --- | --- |
| `model_learning_rate` | `2e-5` to `1e-4` | Log-uniform |
| `model_target_kl` | `0.005` to `0.03` | Uniform |
| `model_batch_size` | `256` or `512` | Categorical |
| `model_sde_xyz_log_std_init` | `log(0.1 / 7.1)` to `-2.6` | Uniform in log SD |
| `model_sde_gripper_log_std_init` | `log(0.35 / 7.1)` to `log(0.75 / 7.1)` | Uniform in log SD |
| `model_transformer_embedding_dim` | 128 to 160, in steps of 4 | Quantized uniform |
| `model_transformer_feedforward_dim` | 512 to 768, inclusive | Integer uniform |
| `model_transformer_layers` | 3 or 4 | Integer uniform |

Embedding widths use multiples of four to remain divisible by the default
four attention heads. Feedforward widths can use every integer in their range.

The noise ranges correspond to approximate initial effective SDs of 0.1–0.53
for XYZ and 0.35–0.75 for the gripper at the reference actor feature norm of
7.1. Rewards, attention heads, and other omitted settings use the
current config defaults. The sweep maximizes logged training success rate and
continues launching trials until stopped. Learning-rate bounds use W&B's
`log_uniform_values`; noise parameters already contain logarithms, so they use
`uniform`. See the [W&B distribution definitions](https://docs.wandb.ai/models/sweeps/sweep-config-keys#distribution-options-for-random-and-bayesian-search).