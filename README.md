# Robot Arm

Shared cube perception and action-policy interfaces for stacking an orange cube
on a blue cube with an SO-101 arm, in MuJoCo simulation and on physical hardware.
The active implementation lives in `src/`, with shared models, buffered camera
input, Cartesian IK, and a paced simulation/hardware control loop. See
[PLAN.md](PLAN.md) for the contracts. Archived code and datasets are not imported
by the new implementation.

## Setup and commands

Use Python 3.12 and the repository's SO-101 model assets in `models/so101/`.

```sh
python -m pip install -r requirements.txt
python -m pytest

PYTHONPATH=src python scripts/generate_vision_data.py --config configs/default.json
PYTHONPATH=src python scripts/train_vision.py --config configs/default.json
PYTHONPATH=src python scripts/train_action.py --config configs/default.json
PYTHONPATH=src python scripts/run_simulation.py --config configs/default.json
```

`configs/default.json` contains every configurable training value. All sections
and fields are required; Pydantic rejects unknown keys and incorrect types.
Paths in this file are relative to the repository root, independent of the
launch directory. Training selects CUDA, MPS, or CPU when `device` is `"auto"`.
The first vision training run downloads the selected ImageNet ResNet-50 weights.
Loading a saved vision checkpoint needs no download.

The default action model uses 384 history tokens, a 160-wide transformer with
4 layers, 4 heads and a 512-wide feedforward network, followed by two 128-unit
action MLP layers. The vision pose MLP has one 2,048-unit hidden layer with ReLU
and `vision_model.dropout: 0.1` (10% dropout after each hidden activation during
training, disabled for validation and inference). Its input
convolution is trainable and starts as an identity filter, followed by fixed
resizing to 224×224. Both camera views share this adapter and frozen ResNet.

Vision loss is `position_loss_weight * XYZ_MSE + orientation_loss_weight * quaternion_MSE`,
averaged over cubes and examples. `training.vision.position_loss_weight` defaults
to `200.0` in the config and `orientation_loss_weight` to `1.0`. The loss function
has the same defaults; training and validation explicitly pass the configured
values. Weights must be nonnegative, with at least one positive. XYZ MSE uses
positions in meters; quaternion MSE normalizes quaternions and accounts for
their sign equivalence.

Training uses a tqdm progress bar across training and validation batches, with
the current epoch, phase, running training loss, and latest completed validation
loss. It starts at epoch 0 with both losses shown as `NaN`. Validation runs at
the configured interval and on the final epoch; its loss stays `NaN` until the
first check, then retains the latest result between checks. The bar also shows
elapsed time, batch rate, and estimated time remaining. Training saves a
single final `.pt` checkpoint, overwriting that checkpoint name. The defaults
are `checkpoints/vision/default.pt` and `checkpoints/action/default.pt`.
Checkpoints include model configuration and all model weights, including the
frozen backbone. Validation here measures held-out tensor loss, not live task
success. These are supervised trainers; they do not run PPO.

Vision training writes a compact log to `logs/default.log` by default. To
choose another path, use `--log-file PATH` or its alias `--log PATH`:

```sh
PYTHONPATH=src python scripts/train_vision.py --log-file logs/vision_training.log
```

The file and parent directories are created if needed; existing logs are
appended to. Each run begins with `epoch=0 train_loss=NaN val_loss=NaN`, then
writes one line per completed epoch containing only `epoch`, `train_loss`, and
`val_loss`. Losses use the configured weights. An epoch without validation
records `val_loss=NaN`; the log never presents an older result as a new
validation measurement. Every line is flushed immediately so another process
can read it while training continues. The terminal retains its progress bar.
Log paths are relative to the working directory unless an absolute path is given.

To measure a saved vision model's position accuracy on the validation set:

```sh
python scripts/diagnostics/vision_position_error.py
python scripts/diagnostics/vision_position_error.py --checkpoint checkpoints/vision/custom.pt
```

The diagnostic reports mean absolute X, Y, and Z error in centimeters for each
cube and both cubes combined. It takes the absolute value of each coordinate
error before averaging over all validation examples, including the final
partial batch. Use `--config PATH` to select another configuration; the default
checkpoint is its `training.vision.checkpoint_path`, normally
`checkpoints/vision/default.pt`. It reuses the validation loader without
RandAugment and runs inference with dropout disabled. No retraining or
`PYTHONPATH` setup is needed.

To split validation loss into its position and orientation contributions:

```sh
python scripts/diagnostics/vision_loss_breakdown.py
python scripts/diagnostics/vision_loss_breakdown.py --checkpoint checkpoints/vision/custom.pt
```

This diagnostic shares the training loss calculation and explicitly applies
the selected config's position and orientation weights. It reports weighted
XYZ and sign-invariant normalized-quaternion MSE contributions, their sum, and
each component's percentage of the sum. Results include each cube and both
combined, averaged over validation examples. The combined total matches
validation with the same weights up to floating-point rounding; older
checkpoints may record losses calculated with different weights. These are
loss contributions, rather than errors in centimeters or degrees. `--config`
and the default checkpoint behave the same as in the position diagnostic.

`run_simulation.py` loads those checkpoints and reads both simulated cameras
through independent 30 FPS production. The command loop runs at 20 Hz until
Ctrl-C, or for `--steps N` ticks. Use `--vision-checkpoint PATH` and
`--action-checkpoint PATH` to select other files. It is a headless control loop;
`scripts/temp.py` remains the scene preview. MuJoCo rendering needs graphics
access even when no viewer window is shown.

## Training data

Vision generation adds UUID-named `.pt` sequence files under `data/vision/train/`
and `data/vision/validation/`. Camera mounts, lens distortion, field of view,
lighting, appearance, cube poses, and arm motion vary. Each sequence keeps its
camera setup fixed and belongs to only one split. The initial generator uses
randomized physical scenes, including resting, elevated/falling, and stacked
cubes. Each sequence independently samples its mode with probabilities of 10%
ground, 10% elevated, 10% stacked, 60% task, and 10% carry. These are configured by
`cube_pose_mode_probabilities` in the same order as `cube_pose_modes`; actual
counts vary between generator runs. The `task` mode uses a scripted
approach/grasp/lift/carry routine with an intentional drop and retry before
placement. The `carry` mode shares the pickup routine, then keeps the gripper
closed while moving the orange cube between random XYZ targets for the rest of
the episode. An accidental drop triggers another pickup attempt. Target ranges
are configured under `environment.vision_data.carry`, in meters for the cube
center: X `[0.25, 0.35]`, Y `[-0.10, 0.10]`, Z `[0.08, 0.13]` by default.
These files train perception;
phase names describe attempted behavior, while separate measured milestone
flags report actual grasp/lift/drop/stack coverage. They are not automatically
accepted as successful action demonstrations. Evaluate coverage and held-out
real camera images before drawing conclusions about deployment accuracy.

Visual randomization is configured under `environment.vision_data`:
`camera_position_jitter` supplies per-axis X/Y/Z bounds in meters, initially
`[0.01, 0.01, 0.01]` (±1 cm on each axis). These are parent-body axes,
so wrist-camera offsets follow the arm. Both cameras independently sample a
vertical field of view from `fov_range: [65.0, 100.0]` once per sequence.
For the wrist camera, the generator adjusts its focal lengths to realize the
sampled field of view while preserving sensor size and lens calibration ratios.
Both views also receive lens distortion, brightness, noise, and blur augmentation.

Each camera independently samples these additional settings once per sequence;
their actual values are saved in `camera_setup`:

| Setting | Initial range | Effect |
| --- | --- | --- |
| `white_balance_gain_range` | `[0.9, 1.1]` for each RGB channel | Mild camera color casts, independent of the scene's light color |
| `gamma_range` | `[0.9, 1.1]` | Applies `output = input ** gamma` to normalized pixels; values above 1 darken midtones |
| `focus_distance_range` | `[0.15, 0.6]` meters | Keeps that optical-axis distance sharp and softens other depths |
| `motion_blur_exposure_range` | `[0.002, 0.012]` seconds | Short shutter exposure when motion blur is selected |

Focus is an approximation using rendered depth: Gaussian blur sigma is
`focus_blur_scale * abs(1 / depth - 1 / focus_distance)`, with initial scale
`0.6` and a `focus_blur_max_sigma` cap of `2.0` pixels. It does not simulate
aperture or bokeh. Zero scale or cap disables it. Motion blur occurs with
`motion_blur_probability: 0.25` per frame and averages `motion_blur_samples: 3`
poses over the exposure, including movement of the wrist camera. Each pose uses
matching depth for focus. These are constant-velocity approximations ending at
the capture timestamp, rendered from private copies without advancing physics
or changing labels. Static scenes skip temporal sampling; exposures are shortened at startup
to avoid sampling before time zero. Probability zero disables motion blur.
Depth and shutter sampling require extra rendering passes.

`light_position_jitter: [0.1, 0.1, 0.1]`
moves the overhead light by ±10 cm in X/Y/Z around `(0, 0, 1.524)` meters,
with a base height of 5 ft above the tabletop. The
overhead light is a positioned spotlight that stays aimed at the enclosed
tabletop's center `(0.20, 0, 0)` after each position offset, changing the
illumination and shadows. Ambient brightness is sampled from
`ambient_intensity_range: [0.2, 0.8]`, independently of diffuse brightness in
`light_intensity_range`. Each light and the headlight receive a random hue with
`light_saturation_range: [0.0, 0.08]`, keeping the tint close to white. Each
brightness is the largest RGB channel after tinting. Scene lights cast shadows
when `shadows_enabled` is true (the default); MuJoCo's headlight does not cast
shadows. Camera calibration, white balance, gamma, focus distance, shutter duration,
and lighting stay fixed for each sequence; brightness, noise, and blur events vary
per frame. Regenerate
data to apply changed settings to images; existing tensor files retain theirs.

Each vision file contains:

| Key | Tensor shape / contents |
|---|---|
| `wrist_images`, `external_images` | `[N, 300, 300, 3]`, RGB `uint8` |
| `joint_positions` | `[N, 6]`, radians |
| `cube_poses` | `[N, 14]`, orange XYZ/wxyz then blue XYZ/wxyz |
| `robot_timestamps`, `wrist_timestamps`, `external_timestamps` | `[N]`, original session timestamps |
| Metadata | Sequence UUID, visual setup, pose mode, phase/milestone records, explicit generation config |

Ground truth is used only for dataset labels. Runtime cube observations always
come from the image model. Images are kept as bytes in memory until minibatch
normalization. The trainer loads all shards into memory before training; reduce
the dataset size if needed. Default generation takes about 75 minutes of control
time (100 sequences × 45 seconds), plus rendering and file-writing overhead.

The vision training loader applies fresh, independent RandAugment to each camera
image on every sample access, using `training.vision.randaugment` in
`configs/default.json` (`magnitude: 5`, `num_ops: 2`). This uses torchvision's
standard operation pool, including rotations, translations, and shearing.
Validation images and pose labels are unchanged. Float image shards are
quantized to bytes for augmentation and then
returned to `[0, 1]`; byte shards stay as bytes until minibatch normalization.
Magnitude ranges from 0 to 30; autocontrast and equalization ignore magnitude.
Set `num_ops` to 0 to disable the training augmentation.

Action training expects **new-schema demonstrations**, one `.pt` dictionary per
episode under `data/action/train/` or `data/action/validation/`:

| Key | Tensor shape / meaning |
|---|---|
| `observations` | `[T, 41]`, observations before each action, ordered below |
| `prev_accepted_targets` | `[T, 3]`, accepted XYZ before that action |
| `actions` | `[T, 4]`, intended action labels at the same timestamp |

Split complete episodes between training and validation. For perception-matched
action training, record observations built from the vision estimates. The loader
shifts actions to construct previous-action inputs; it never uses the current
label as the current input. It materializes right-padded 384-token windows with
stride 256, causal attention, padding masks, and per-token loss masks. Overlap
is context for later windows; each label contributes to loss once. Existing
archived demonstrations/checkpoints are not converted automatically.

## Runtime contracts and hardware boundary

The application owns one `SimulationSession`, which shares its environment
between telemetry, cameras, and command execution. Reset clears camera buffers
atomically with physics and then resets model history. Camera errors are latched
and reported on the control thread. Missing frames temporarily retain the last
joint targets; a frame or continuous read failure older than one second stops
the run with an exception. Only the command backend advances simulation.

The physical classes accept one shared `SO101Connection` adapter implementing
`read_joint_state()` and `send_joint_targets()`, in the documented joint order
and radian units. They do not open or calibrate a robot. Mapping a real LeRobot
connection's telemetry/commands to this interface remains a hardware integration
step: joint signs, offsets, velocities, and startup targets must be established
for the actual arm. Camera workers accept externally opened OpenCV cameras
already configured for 300×300 at 30 FPS. No live hardware was tested here.

## Vision model

The vision model consumes the latest wrist RGB image, latest external-camera
RGB image, and latest six joint positions. Each image is 300 × 300 pixels with
three RGB channels. Buffered images remain `uint8` in `[0, 255]`; model inputs
convert to `float32` and divide by `255.0`, giving pixels in `[0, 1]`.

Joint inputs are `float32` radians in this order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Each input uses its latest available reading, without matching or interpolating
capture times. Camera setup parameters are used for training-data generation,
rather than supplied as runtime model inputs.

The model predicts both cube poses in robot-base coordinates:

| Output | Values | Count |
|---|---|---:|
| Orange position | XYZ, meters | 3 |
| Orange orientation | Quaternion `(w, x, y, z)` | 4 |
| Blue position | XYZ, meters | 3 |
| Blue orientation | Quaternion `(w, x, y, z)` | 4 |
| **Total** | **2 × (3 + 4)** | **14** |

Linear velocity is derived by `CubeStateProvider` from successive position
estimates and elapsed time; it is included in the action-policy observation,
but is not a vision-network output.

## Action policy

The policy receives `float32` observations in the following source-defined
order. Joint groups follow the six-joint order above. XYZ positions and linear
velocities use the robot-base frame; quaternions use `(w, x, y, z)`.

| Index range | Input | Units | Count |
|---|---|---|---:|
| `[0:6]` | Joint positions | Radians | 6 |
| `[6:12]` | Joint velocities | Radians/second | 6 |
| `[12:18]` | Actuator control targets | Radians | 6 |
| `[18:21]` | Gripper reference (`gripperframe`) XYZ | Meters | 3 |
| `[21:24]` | Orange XYZ | Meters | 3 |
| `[24:28]` | Orange quaternion `(w, x, y, z)` | Unitless | 4 |
| `[28:31]` | Orange linear velocity | Meters/second | 3 |
| `[31:34]` | Blue XYZ | Meters | 3 |
| `[34:38]` | Blue quaternion `(w, x, y, z)` | Unitless | 4 |
| `[38:41]` | Blue linear velocity | Meters/second | 3 |
| **Observation total** | | | **41** |
| `[41:45]` | Previous issued policy action `[dx, dy, dz, gripper]` | Normalized | 4 |
| `[45:48]` | `prev_accepted_target` XYZ | Meters | 3 |
| **History-token total** | | | **48** |

`prev_accepted_target` is the last Cartesian destination successfully issued
after solving inverse kinematics. It starts at the measured/forward-kinematics
gripper position before the first command.

The policy keeps up to 384 history tokens, with separate `valid_mask` and
`episode_start_mask` arrays. Using only the latest readings for each vision
estimate preserves this action-policy history. The policy predicts a normalized
four-value action `[dx, dy, dz, gripper]`, which the Cartesian controller converts
to joint targets.
