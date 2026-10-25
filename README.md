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
python scripts/generate_action_data.py --config configs/default.json
PYTHONPATH=src python scripts/train_action.py --config configs/default.json
PYTHONPATH=src python scripts/run_simulation.py --config configs/default.json
```

`configs/default.json` contains every configurable training value. All sections
and fields are required; Pydantic rejects unknown keys and incorrect types.
Paths in this file are relative to the repository root, independent of the
launch directory. Training selects CUDA, MPS, or CPU when `device` is `"auto"`.
The first vision training run downloads the selected ImageNet ResNet-34 weights
(`IMAGENET1K_V1` in the default config).
Loading a saved vision checkpoint needs no download.
Action-data generation requires a compatible trained vision checkpoint; by
default it uses `training.vision.checkpoint_path`.

The default action model uses 384 history tokens of 44 values each, a learned
`Linear(44, 160)` input projection, and a 160-wide transformer with
4 layers, 4 heads and a 512-wide feedforward network, followed by two 128-unit
action MLP layers. `action_policy.transformer.dropout: 0.1` enables 10% dropout
inside the transformer during training, disabled for validation and inference.
The vision pose MLP has one 2,048-unit hidden layer with ReLU
and `vision_model.dropout: 0.1` (10% dropout after each hidden activation during
training, disabled for validation and inference). The external-camera input
is 300×300. Its trainable 3×3 convolution starts as an identity filter, followed
by fixed resizing to 224×224,
ImageNet normalization, and ResNet-34 with its classifier removed. All ResNet
weights are fine-tuned, and its BatchNorm statistics update during training.

The wrist camera supplies 128×128 images to a separate small CNN, configured through
`vision_model.wrist_cnn`: `layers: 2`, `channels: [16, 16]`, and
`embedding_dim: 128`. Each layer uses a 3×3 convolution with stride 2, padding 1,
and ReLU. Adaptive average pooling to 4×4, flattening, and a linear projection
produce the wrist embedding. The channel list must have one positive entry per
layer. The pose MLP concatenates 128 wrist-camera features, 512 external-camera
features, and six joint positions, giving 646 inputs with the default config.

Vision loss is `position_loss_weight * XYZ_MSE + orientation_loss_weight * yaw_pair_MSE`,
averaged over cubes and examples. `training.vision.position_loss_weight` defaults
to `200.0` in the config and `orientation_loss_weight` to `1.0`. The loss function
has the same defaults; training and validation explicitly pass the configured
values. Weights must be nonnegative, with at least one positive. XYZ MSE uses
positions in meters. Yaw loss normalizes the predicted and target
`[cos(4θ), sin(4θ)]` pairs and averages their two squared component errors.
This circular representation makes yaw periodic over 90° for uniformly colored,
unmarked cubes: 89.9° and 0.01° are only 0.11° apart across the boundary.
Distinct face markings would require reconsidering that symmetry assumption.

Training, validation, and the loss diagnostic share this calculation. New vision
labels contain ten values, while the loader also accepts existing 14-value
quaternion labels and converts them in saved sequence order using the same
face-selection and leveling rule, retaining the previous face only for ties.
Compatible image datasets can therefore be
reused. The vision model's changed output head requires retraining; previous
quaternion-loss values are not directly comparable to the circular yaw loss.

Vision training uses batch size 16 by default. Its learning rate starts at
`0.0001` and decreases when validation loss plateaus, using the settings in
`training.vision.lr_scheduler`. The default `reduce_on_plateau` scheduler halves
the rate after two consecutive validation checks without more than a 0.1%
improvement over its best loss, down to a minimum of `0.000001`. Its `patience: 1`
tolerates one non-improving check; the second triggers the reduction. With
validation every 10 epochs, those two checks normally span 20 epochs after the
last qualifying improvement. The scheduler uses only newly calculated weighted
validation loss, including the final validation check. Reductions apply to the
following epoch. Set `lr_scheduler` to `null` to keep a constant rate; older
configs without the field also retain constant-rate behavior. Action training
keeps its existing learning rate.

Training uses a tqdm progress bar across training and validation batches, with
the current epoch, phase, running training loss, and latest completed validation
loss. Vision training also shows the latest mean validation position error in
meters; action training shows rollout success percentage. These metrics start
as `NaN` at epoch 0. Validation runs at the configured
interval (every 10 epochs by default) and on the final epoch; validation metrics
stay `NaN` until the first check, then retain the latest result between checks.
The bar also shows
elapsed time, batch rate, and estimated time remaining. Training saves a
single final `.pt` checkpoint, overwriting that checkpoint name. The defaults
are `checkpoints/vision/default.pt` and `checkpoints/action/default.pt`.
Checkpoints include model configuration and all model weights. Older shared
ResNet-50 checkpoints are incompatible with this architecture; train a new
vision checkpoint after regenerating the dataset. Action validation measures
both held-out tensor loss and simulated stacking success, as described below.
Vision and action checkpoints now use format version 3. Vision versions 1 and 2
predict quaternions and require retraining for the ten-output head. Action
version 1 used 48-value tokens, while version 2 used 5% face-switch hysteresis;
both require retraining for the new direct-yaw convention. The action model
still receives 44-value tokens. Existing 41-value action demonstration tensors
remain reusable through in-memory conversion.
These are supervised trainers; they do not run PPO.

Vision training writes a compact log to `logs/default.log` by default. To
choose another path, use `--log-file PATH` or its alias `--log PATH`:

```sh
PYTHONPATH=src python scripts/train_vision.py --log-file logs/vision_training.log
```

The file and parent directories are created if needed; existing logs are
appended to. Each run begins with
`epoch=0 train_loss=NaN val_loss=NaN val_position_error_m=NaN`, plus
`learning_rate` and `next_learning_rate` set to the initial rate, then writes one
line per completed epoch with those six fields. `learning_rate` records the
rate used during that epoch; `next_learning_rate` records the rate after any
validation-driven reduction, to be used by the next epoch. Losses use the configured
weights. `val_position_error_m` is the arithmetic mean of the two cubes' 3-D
Euclidean position errors per example, averaged over all validation examples:
`mean_i((||pred_orange_i - target_orange_i||₂ + ||pred_blue_i - target_blue_i||₂) / 2)`.
It uses positions in meters, without loss weighting or squaring the distances,
and is computed from the existing validation predictions without an extra model
pass. An epoch without validation records both validation fields as `NaN`; the
log never presents an older result as a new validation measurement. Every line
is flushed immediately so another process can read it while training continues.
Existing log lines are unchanged; an already-running experiment must be restarted
to use updated logging code. The terminal retains its progress bar.
Log paths are relative to the working directory unless an absolute path is given.

To measure a saved vision model's position accuracy on the validation set:

```sh
python scripts/diagnostics/vision_position_error.py
python scripts/diagnostics/vision_position_error.py --checkpoint checkpoints/vision/custom.pt
```

The diagnostic reports mean absolute X, Y, and Z error in centimeters for each
cube and both cubes combined. It takes the absolute value of each coordinate
error before averaging over all saved validation examples, including the final
partial batch. Use `--config PATH` to select another configuration; the default
checkpoint is its `training.vision.checkpoint_path`, normally
`checkpoints/vision/default.pt`. It loads every row from the saved validation
files, as training-time validation does. It reuses the validation loader without
RandAugment and runs inference with dropout disabled. No retraining or
`PYTHONPATH` setup is needed.

To split validation loss into its position and orientation contributions:

```sh
python scripts/diagnostics/vision_loss_breakdown.py
python scripts/diagnostics/vision_loss_breakdown.py --checkpoint checkpoints/vision/custom.pt
```

This diagnostic shares the training loss calculation and explicitly applies
the selected config's position and orientation weights. It reports weighted
XYZ and normalized circular-yaw-pair MSE contributions, their sum, and
each component's percentage of the sum. Results include each cube and both
combined, averaged over validation examples. The combined total matches
validation with the same weights and orientation metric up to floating-point
rounding; older checkpoints may record losses calculated with different
weights or without cube symmetry. These are
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
randomized resting and stacked scenes plus scripted manipulation attempts.
Each sequence independently samples its mode with probabilities of 30% ground,
10% stacked, and 60% task. These are configured by
`cube_pose_mode_probabilities` in the same order as `cube_pose_modes`; actual
counts vary between generator runs. The `task` mode uses a scripted
approach/grasp/lift/carry routine with an intentional drop and retry before
placement. The default mix excludes the standalone `elevated` and `carry` modes.

For uniformly sampled **gripper-frame** targets, use the separate generator:

```bash
.venv/bin/python scripts/generate_vision_uniform_data.py --samples 5000
```

It samples X/Y independently from the vision cube-spawn bounds and Z uniformly
from 3–8 cm. The arm physically picks up orange when needed, retains its grasp
between successful iterations, and moves the gripper reference to each target.
Saved labels always describe the actual simulated cubes. Blue is resampled on
the table each iteration, with clearance from orange and the destination.
Both cameras are resampled from their original mounts every iteration, using
the existing camera/lens/effect distributions without accumulating drift.
Each new scene uses the same surface, lighting, texture, and paper-color
randomization as the original generator; those settings persist until reset.

`--samples` counts successfully saved examples (default 5000). Each example has
one UUID-named `.pt` file with leading dimension 1 and the existing tensor
schema. Files append under the configured vision `train`/`validation` folders;
`--output-dir` can select a separate dataset. The configured `train_fraction`
determines split counts, and a fresh scene is created at the split boundary so
a persistent scene cannot appear in both splits. Scene/run IDs, requested and
actual gripper positions, camera setup, and simulator labels are saved for
diagnostics. A target is uniform at sampling time; rejected moves can bias the
distribution of successful saved samples.

Exceptions (including IK, pickup, movement, and rendering failures) are logged,
the scene is recreated at the end of that attempt, and another target is
sampled. The JSON-lines event log defaults to `logs/vision-workspace-<run>.log`;
override it with `--log-file`. It includes every saved UUID/path and failed
attempt with diagnostics. Console messages use `tqdm.write`; progress-bar output
is never redirected into the log. The run stops after 20 consecutive sample or
scene-creation failures, or `--max-attempts` (default 10 times the sample count),
preserving completed files. Simulation has no artificial pacing delay.

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

The enclosure uses construction-paper base colors shared by vision-data
generation, rollout validation, and the demo. In a top-down view with +X to the
right and +Y up, the top/bottom walls (`wall_y_min`, `wall_y_max`) are green
`#4E8A54`, the left/right walls (`wall_x_min`, `wall_x_max`) are red `#B64545`,
and the tabletop is brown `#926849`. The scene XML stores these RGB bytes
divided by 255 as the single source for the palette. These are representative
matte construction-paper shades, not measured manufacturer color standards;
the color families follow common assortments such as
[Crayola's construction paper](https://www.crayola.com/products/coloring-books-and-paper/bulk-construction-paper-480-count-990013).

Only vision-data generation varies the paper colors. Each surface independently
samples a bounded triangular distribution concentrated at its base, once per
sequence, using `environment.vision_data.paper_color_variation`:

| Setting | Default maximum deviation | Effect |
| --- | --- | --- |
| `hue_shift` | `0.01` turn (3.6°) | Small dye/hue differences |
| `saturation_scale` | `0.10` (±10%) | More or less gray/muted |
| `brightness_scale` | `0.12` (±12%) | Lighter/darker paper |
| `warmth` | `0.025` (±2.5%) | Opposite red/blue channel gains |

Hue wraps around; saturation, value, and final RGB are bounded to `[0, 1]`.
Setting all four deviations to zero preserves the base colors exactly.
The generator records the bases, sampled RGB, and variation factors under
`camera_setup["paper_colors"]`. Colors stay fixed throughout each sequence;
the existing lighting, exposure, and white-balance augmentation still operates
separately. The old `wall_brightness_range` and `wall_saturation_range` settings
have been replaced by this base-relative variation.

Validation and the demo keep the exact base material colors with no paper-color
randomization. Ordinary lighting and shadows still affect the rendered pixels.
The XML's `construction_paper` material supplies fixed defaults for validation:
`specular=0.2`, `shininess=0.1`, and `reflectance=0`. During data
generation, code creates independent materials for the walls and tabletop and
overrides those properties using the configured distributions described below.
Cube identity colors are unchanged. Existing datasets and MP4s retain their old appearance;
generate fresh vision data and retrain the vision model to learn the new setup,
then rerun the demo to render the new environment. The generator appends to its
data directory, so select a fresh `training.vision.data_dir` when replacing the
old palette rather than mixing old and new images. Both generated vision-data
splits use the color variation; fixed bases refer to action rollout validation
and the demo.

Material appearance is sampled independently once per sequence. The 12 settings
`specular_mean`, `specular_variance`, `specular_min`, `specular_max`,
`shininess_mean`, `shininess_variance`, `shininess_min`, `shininess_max`,
`reflectance_mean`, `reflectance_variance`, `reflectance_min`, and
`reflectance_max` also live under `environment.vision_data`. Specular and
shininess each default to mean `0.65`, variance `0.55`, minimum `0.1`, and
maximum `0.8`. Reflectance defaults to mean `0.2`, variance `0.33`, minimum `0`,
and maximum `0.7`.
`draw_clipped_val_from_gaussian` uses Gaussian rejection sampling: draw with
standard deviation `sqrt(variance)` and retry until the value lies within the
inclusive bounds. Values are not clamped to the endpoints; the configured mean
and variance describe the Gaussian before rejection.

The generator randomizes specular and shininess for each existing named arm
material; parts sharing a material share the sampled values. It creates seven
independent materials in memory for the two cubes, four walls, and table. All
seven surfaces receive independently sampled specular, shininess, and
reflectance values from the config. Generation never writes these variations
back to the source XML. Sampled settings are saved in
`camera_setup["materials"]` and stay fixed throughout the sequence. MuJoCo's
native renderer limits mirror reflections to the first eligible surface; the
tabletop can therefore appear reflective during generation. Specular and
shininess affect the arm, cubes, walls, and tabletop. Generate new data to use
these settings; existing images and
running training jobs are unchanged.

Each newly generated vision file contains five paired examples by default
(`N = 5`), controlled by `environment.vision_data.samples_per_sequence`:

| Key | Tensor shape / contents |
|---|---|
| `wrist_images` | `[N, 128, 128, 3]`, RGB `uint8` |
| `external_images` | `[N, 300, 300, 3]`, RGB `uint8` |
| `joint_positions` | `[N, 6]`, radians |
| `cube_poses` | `[N, 10]`, orange XYZ/cos(4θ)/sin(4θ), then blue XYZ/cos(4θ)/sin(4θ) |
| `robot_timestamps`, `wrist_timestamps`, `external_timestamps` | `[N]`, matching simulation timestamps |
| Metadata | Sequence UUID, visual setup, pose mode, `pose_representation`, phase/milestone records, explicit generation config |

For vision generation, ground truth supplies the dataset labels. Yaw labels
select the most horizontal face, retaining the previous face only when vertical
scores tie. Independent encoders update at every control tick, including
unsaved frames, and reset for each generated rollout. Runtime cube
observations always come from the image model. Images are kept as bytes in memory until minibatch
normalization. Before collecting a rollout, the generator randomly selects
`environment.vision_data.samples_per_sequence: 5` distinct observation times
from the candidate steps defined by `sample_interval_steps` (once per simulated
second with the default settings). Selection uses the rollout's random seed,
and the selected samples are collected in chronological order across the full
rollout. At each selected step, the generator synchronously renders both cameras
from the same simulation state and saves their images with matching joints,
pose labels, simulation timestamps, and per-sample metadata. It uses no camera
worker or frame buffer; cameras render only for those five observations, with
extra depth and exposure passes for focus and motion blur as configured. The
same count applies to training and validation. A rollout with fewer candidate
times than the requested count is rejected before collection.

Training, validation, and diagnostic loaders load every saved row without
subsampling, resizing, or upsampling. They require 128×128 wrist images and
300×300 external images. The training DataLoader still shuffles sample order
each epoch. Existing files are not rewritten: datasets containing 128×128
external images must be regenerated before training, diagnostics, or previews.
Restart any already-running generator processes to load the new capture sizes.
Older files with 45 rows contribute all 45 rows when their image sizes match.
Default generation adds 1,200 sequences per run: 1,000 training and 200 validation.
The tqdm bar counts saved paired samples, advancing by five after each rollout
is saved (6,000 samples total by default). Each rollout's UUID, split, mode, and
available scenario statistics are printed above the bar using `tqdm.write()`.
Saving five paired examples per sequence gives 5,000 training
and 1,000 validation examples per run. Each rollout still runs for 45 simulated
seconds, sending a command every 0.05 simulated seconds (20 Hz, 900 control
steps). Generation does not sleep to keep pace with wall-clock time. Its elapsed
time depends on simulation, rendering, and file writing, so a rollout can take
less or more than 45 seconds. Runtime inference retains its separate camera
production and latest-frame buffers described above.

The vision training loader applies fresh, independent RandAugment to each camera
image on every sample access, using `training.vision.randaugment` in
`configs/default.json` (`magnitude: 7`, `num_ops: 2`). This uses torchvision's
standard operation pool with `Solarize` excluded, including rotations,
translations, and shearing. The sample-preview scripts use this same loader.
Validation images and pose labels are unchanged. Float image shards are
quantized to bytes for augmentation and then
returned to `[0, 1]`; byte shards stay as bytes until minibatch normalization.
Magnitude ranges from 0 to 30; autocontrast and equalization ignore magnitude.
Set `num_ops` to 0 to disable the training augmentation.

`scripts/generate_action_data.py` collects successful orange-on-blue stacking
demonstrations with a scripted teacher adapted from the legacy generator.
The script defaults to adding exactly 5,000 successful training episodes and
1,000 successful validation episodes per run, set by
`NUMBER_OF_TRAINING_EXAMPLES = 5000` and
`NUMBER_OF_VALIDATION_EXAMPLES = 1000`. Existing data is preserved. Override
either count with `--training-examples N` or `--validation-examples N`; an omitted
flag keeps its corresponding script default:

```sh
.venv/bin/python scripts/generate_action_data.py --config configs/default.json --training-examples 10 --validation-examples 2
```

The script runs directly without setting `PYTHONPATH`. Use
`--vision-checkpoint PATH` to override `training.vision.checkpoint_path`, and
`--device cpu|mps|cuda|auto` to override `training.vision.device`. It loads the
vision model once, freezes its parameters, and keeps it in evaluation mode.
Missing or incompatible checkpoints, including older 14-output vision models,
fail before collection; generation never falls back to exact simulator inputs.

The output directory is `training.action.data_dir`, normally `data/action`, with
complete episodes saved under `train/` and `validation/`. The compatible
`--examples N` option instead specifies a total count and splits it using
`environment.action_data.train_fraction: 0.8`. That ratio applies only to total-
count mode; explicit split counts, including the defaults, determine their own
allocation. `--examples` cannot be combined with either explicit split flag.
Each UUID-named `.pt` dictionary contains the new model's format:

| Key | Tensor shape / meaning |
|---|---|
| `observations` | `[T, 37]`, vision-based policy observations before each action, ordered below |
| `prev_accepted_targets` | `[T, 3]`, accepted XYZ before that action |
| `actions` | `[T, 4]`, normalized intended teacher actions at the same timestamp |
| Metadata | Dataset schema version 2, `observation_source: "vision_model"`, vision checkpoint path, episode UUID/seed, generation settings, stage counts, drop/recovery records, and success outcome |

The three tensors are CPU `float32` and share the same length `T`; every row is
aligned to the state before its action. Two synchronous camera renderers are
reused across all attempts. Before each action, the vision model processes the
current camera images and joint positions to produce ten cube features: XYZ
and normalized `cos(4θ), sin(4θ)` for each cube. Six linear-velocity features
come from successive estimated positions; the remaining 21 values are robot
telemetry. Images are consumed during collection and are not saved in these
episode files. Velocity history resets for each new attempt and remains intact
through partial teacher recovery. Both saved splits therefore use the same
learned-perception path as policy rollout validation.

The scripted teacher still uses true simulator state to choose intended action
labels, detect grasps, recover dropped cubes, and judge success. Those exact cube
measurements do not enter the newly saved observations.

All scene sampling, teacher, intentional-drop, recovery, and success settings
are explicit under `environment.action_data`. Initially, cubes spawn on the
table with identity orientation, X in `[0.25, 0.35]` m, Y in `[-0.1, 0.1]` m,
and at least `0.08` m center separation. The open gripper starts around
`[0.20, 0, 0.05]` m with independently sampled ±4 cm X/Y offsets.

While orange is held, each eligible control tick has a
`drop_disturbance.probability: 0.002` chance of an intentional drop. A thin
command layer after teacher IK holds the measured arm pose for
`stop_seconds: 0.25`, then forces the gripper open for `release_seconds: 0.25` while keeping
the arm still. An active disturbance cannot retrigger. Saved actions remain the
teacher's intended commands; observations and accepted-target history reflect
the commands actually sent to the simulation.

The teacher detects a lost grasp after two control ticks, or retries an unheld
pickup more than five simulated seconds after reaching its waypoint. This
partial reset clears teacher progress while preserving physics, elapsed time,
and every recorded action. It pauses with the claw open, waits for the cube to
settle, and recalculates the pickup waypoint 8 cm above orange's current position.
Deliberate placement release is excluded from drop recovery.

Each attempt has the shared `environment.episode_seconds` budget of 45 simulated
seconds, including recovery (900 control ticks at 20 Hz). Physics advances
without wall-clock waiting. Only released, stable successful stacks are saved;
ordinary failed attempts are discarded, with an attempt budget of ten per
requested example. Required IK failures stop generation and save JSON
diagnostics. Successful episodes are published atomically without replacing
existing files; the UUID supplies the episode's random seed. The tqdm bar counts
committed episodes, and status messages use `tqdm.write()`.

The action training loader uses new 37-value observations directly. It also
accepts existing 41-value ground-truth demonstrations, converting their raw
quaternions using the same cube-yaw conversion as the vision labels. Legacy
conversion covers each full episode before splitting windows, so overlapping
timesteps contain identical processed features. It
shifts actions to construct previous-action inputs; it never uses the current
label as the current input. It materializes right-padded 384-token windows with
stride 256, causal attention, padding masks, and per-token loss masks. Overlap
is context for later windows; each label contributes to loss once. Current
41-value demonstration files need no conversion on disk and can coexist with
new files, but their exact simulator inputs remain different from learned
vision estimates. Archived 49-value
demonstrations and older action checkpoints are not converted automatically.

Every action validation check also runs 200 simulated episodes using the current
action model and the vision checkpoint at `training.vision.checkpoint_path`.
This includes the final epoch, without repeating a check when that epoch already
falls on the validation interval. The vision model is loaded once before
training, kept in evaluation mode, and frozen; only the action model is optimized.
A missing or incompatible vision checkpoint stops training before its first
epoch. Rollout validation therefore needs enough memory for both models.

`training.action.rollout_validation` sets the episode count, scene seed, arm
start distribution, cube spawn ranges, minimum separation, and success
tolerances. The defaults use seed 43, cube X positions in `[0.25, 0.35]` meters,
Y positions in `[-0.1, 0.1]` meters,
at least `0.08` meters between cube centers, and axis-aligned cubes with identity
quaternions. These cube spawn settings match action data generation. Each
validation check reuses the same seeded starting scenes so its success
percentage can be compared with earlier checks. Each episode starts the open
gripper at a sampled Cartesian home centered on `start_position: [0.20, 0, 0.05]`
meters with `start_position_half_range: [0.04, 0.04, 0]`: independently sampled
±4 cm X/Y offsets and fixed Z. These fields live under
`training.action.rollout_validation`. The reset directly initializes the arm
using the same IK and controller constraints as demonstrations, without
advancing simulation or recording policy-history steps. Both cubes, perception
history, policy history, and controller state also reset for each episode.

Rollouts synchronously render the 128×128 wrist and 300×300 external cameras.
The learned vision model estimates cube poses from those images and joint
positions; the action policy receives the resulting observations and its own
action history. Simulation ground truth is used only to score success. The
controller and physics advance one control interval per prediction without
wall-clock pacing or artificial sleeps. An episode lasts at most
`environment.episode_seconds` (45 simulated seconds by default, or 900 control
ticks at 20 Hz). Rendering and model inference still take actual computation
time.

A successful stack requires the orange cube to contact the blue cube, the blue
cube to remain supported by the table, and neither cube to contact the gripper.
For one continuous simulated second, checked at each control tick, the cube
centers must align horizontally within 1 cm and vertically within 5 mm of the
stacked height, with each cube's
linear speed at most 1 cm/s and angular speed at most 0.05 rad/s. These thresholds
are configurable under `rollout_validation.success`. An IK failure ends that
episode as a failure; a timeout also counts as failure. Unexpected simulator,
rendering, or checkpoint errors propagate instead of silently reducing the
number of attempts.

The reported success rate is `100 * successful episodes / attempted episodes`,
separate from the held-out action MSE. Returned validation metrics include
`validation_success_rate_percent`, `validation_rollouts`, `validation_successes`,
`validation_ik_failures`, and `validation_timeouts`; the final checkpoint also
stores the last rollout results in `validation_rollout_metrics`.

Run rollout validation independently of training with a saved action checkpoint:

```sh
python scripts/evaluate_action_rollouts.py --checkpoint checkpoints/action/default.pt --workers 2
```

The script defaults to two spawned processes and `configs/default.json`; no
`PYTHONPATH` setup is needed. Each process loads the selected action checkpoint
and the configured vision checkpoint, then runs
`floor(training.action.rollout_validation.episodes / workers)` episodes. With
200 configured episodes, two workers run 100 each; three run 66 each, for 198
total. The remainder is skipped and reported. Config changes stay in memory.
Workers use distinct seeds (`rollout_validation.seed + worker_index`), so a
fixed worker count gives repeatable scenes without duplicating them across
workers. Each worker shows episode progress; the final report combines successes,
IK failures, timeouts, and elapsed wall time across all attempted episodes.
This evaluates rollout success only, without loading demonstration datasets or
calculating validation MSE. `--config PATH` selects an alternative configuration.

Save a validation demo with the legacy 3-row, 4-column layout:

```sh
.venv/bin/python scripts/demo.py --output stack_demo.mp4
# Use a different saved action checkpoint:
.venv/bin/python scripts/demo.py --checkpoint checkpoints/action/custom.pt --output custom_demo.mp4
```

This writes a 1280×720 H.264 MP4 with twelve 320×240 panels. Each panel shows
the external camera, fitted without stretching or cropping; the policy still
uses both wrist and external images through the current learned-vision
validation loop. The outline is yellow until the released stack passes the
configured stable-success check, then green. Timed-out or IK-failed panels stay
yellow, and all completed panels hold their last frame through the end.

The action and vision weights default to `training.action.checkpoint_path` and
`training.vision.checkpoint_path` in `configs/default.json` (normally
`checkpoints/action/default.pt` and `checkpoints/vision/default.pt`). Use
`--checkpoint PATH` (also `--action-checkpoint` or `--weights`) and
`--vision-checkpoint PATH` to override them. No `PYTHONPATH` setup or viewer is
needed. FFmpeg must be installed on `PATH` (`brew install ffmpeg` on macOS).

The demo always records twelve attempts, using seeds `rollout_validation.seed`
through `seed + 11`, independent of worker count. Two spawned workers share the
work by default; `--workers 1` reduces memory use. Each worker loads its models
once. Playback runs at the configured control rate (20 FPS by default), with
an initial frame followed by up to `environment.episode_seconds` of simulation
(45 seconds by default). `--episode-seconds 5` makes a short preview;
`--seed N`, `--device cpu`, and `--config PATH` are also available. These overrides
do not edit the configuration file. Temporary videos are cleaned up, and an
existing output is replaced only after the complete grid has encoded.

Action training appends an epoch log to `logs/action.log`. Override it with
`--log-file PATH` or `--log PATH`, for example:

```sh
PYTHONPATH=src python scripts/train_action.py --log-file logs/action_training.log
```

Each completed epoch records `epoch`, `train_loss`, `val_loss`, and
`val_success_rate_pct`. Validation fields are `NaN` at epoch 0 and on epochs
without validation; the progress display retains the latest measured results
between checks. Log lines are flushed immediately and existing logs are
preserved. A path is relative to the working directory unless it is absolute.

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
already configured for 128×128 wrist images and 300×300 external images at
30 FPS. No live hardware was tested here.

## Vision model

The vision model consumes the latest wrist RGB image, latest external-camera
RGB image, and latest six joint positions. Wrist images are 128×128 pixels;
external images are captured and saved at 300×300 pixels. Both have three RGB
channels. Buffered images remain `uint8` in `[0, 255]`; model inputs
convert to `float32` and divide by `255.0`, giving pixels in `[0, 1]`.

Joint inputs are `float32` radians in this order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Each input uses its latest available reading, without matching or interpolating
capture times. Camera setup parameters are used for training-data generation,
rather than supplied as runtime model inputs.

The model predicts both cubes' positions and circular yaw features in robot-base
coordinates:

| Output | Values | Count |
|---|---|---:|
| Orange position | XYZ, meters | 3 |
| Orange yaw | `cos(4θ), sin(4θ)` | 2 |
| Blue position | XYZ, meters | 3 |
| Blue yaw | `cos(4θ), sin(4θ)` | 2 |
| **Total** | **2 × (3 + 2)** | **10** |

`CubeStateProvider` normalizes each predicted yaw pair with the same `1e-6`
denominator floor used by the loss before passing it directly to the action
model. Nondegenerate pairs have unit length; an exactly zero pair remains zero.
It does not reconstruct a quaternion
or perform temporal face selection. Positive θ is a counterclockwise turn
about world +Z when viewed from above; `atan2(sin(4θ), cos(4θ)) / 4`, wrapped
into `[0°, 90°)`, recovers an angle for display.

Linear velocity is derived by `CubeStateProvider` from successive position
estimates and elapsed time; it is included in the action-policy observation,
but is not a vision-network output.

## Action policy

New action demonstrations save the 37-value `float32` policy observations
listed below, with learned cube positions and yaw pairs. Existing ground-truth
demonstrations retain the following 41-value layout, including simulator
quaternions, and remain loadable.
Joint groups follow the six-joint order above. XYZ
positions and linear velocities use the robot-base frame; quaternions use
`(w, x, y, z)`.

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
| **Raw observation total** | | | **41** |

When loading legacy action observations or legacy vision labels, shared conversion
replaces each four-value quaternion with two yaw features, `cos(4θ)` and `sin(4θ)`. These
features treat rotations separated by 90° as equivalent for an unmarked cube.
The conversion normalizes the quaternion and independently chooses the cube axis
with the largest absolute Z component (the most horizontal face). It selects
that normal's upward-facing side and applies the smallest rotation that levels
the normal to world +Z.
It then measures an in-plane cube edge's yaw in the leveled frame. Ties use
the previously selected face when its vertical score ties for best within
`1e-6`. The face is associated by its world-space normal, so changing quaternion
sign or relabeling equivalent cube faces does not change that association.
With no prior face, a world-space geometric tie-break gives a deterministic
choice. A clearly more horizontal face always wins immediately.

The earlier 5% face-switch hysteresis and its configuration have been removed;
history matters only when resolving a face-selection tie in label creation or
legacy-data conversion. Legacy vision shards have only their saved samples,
so conversion cannot recover intervening unsaved face history. The per-frame
vision model learns the resulting labels but cannot itself enforce that temporal
tie rule. Its two predicted features enter the action observation directly after
this guarded normalization. Policy history resets at actual episode boundaries;
teacher recovery preserves that history.

New saved observations and the action-model input use this layout:

| Index range | Input | Units | Count |
|---|---|---|---:|
| `[0:21]` | Joint positions, velocities, control targets, and gripper XYZ, as above | As above | 21 |
| `[21:24]` | Orange XYZ | Meters | 3 |
| `[24:26]` | Orange `cos(4θ), sin(4θ)` | Unitless | 2 |
| `[26:29]` | Orange linear velocity | Meters/second | 3 |
| `[29:32]` | Blue XYZ | Meters | 3 |
| `[32:34]` | Blue `cos(4θ), sin(4θ)` | Unitless | 2 |
| `[34:37]` | Blue linear velocity | Meters/second | 3 |
| **Policy observation total** | | | **37** |
| `[37:41]` | Previous issued policy action `[dx, dy, dz, gripper]` | Normalized | 4 |
| `[41:44]` | `prev_accepted_target` XYZ | Meters | 3 |
| **History-token total** | | | **44** |

`prev_accepted_target` is the last Cartesian destination successfully issued
after solving inverse kinematics. It starts at the measured/forward-kinematics
gripper position before the first command.

The trainable `Linear(44, 160)` layer maps each token into the transformer's
160-dimensional embedding space before adding position and episode-start
embeddings. The policy keeps up to 384 history tokens, with separate `valid_mask` and
`episode_start_mask` arrays. Using only the latest readings for each vision
estimate preserves this action-policy history. The policy predicts a normalized
four-value action `[dx, dy, dz, gripper]`, which the Cartesian controller converts
to joint targets.
