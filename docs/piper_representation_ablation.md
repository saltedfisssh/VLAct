# Piper / Piper-X representation experiments in VLAct

This implementation ports the data and deployment conventions from
`/kpfs_ssd/data/wzy/kai0/docs/piper_representation_ablation.md`,
`piper_x_native.md`, and `piper_x_eef_aligned_s2_pair.md` to VLAct's
**QwenPI_v4**. It is a new model experiment, not a reproduction of pi05 losses
or a reuse of pi05 checkpoints. See [deployment](piper_deployment.md),
[Piper-X alignment](piper_x_eef_aligned_s2_pair.md), [mixed training](piper_mixed_training.md), and
[validation records](piper_validation.md).

## Experiment matrix

| Registered `data_mix` | Source | Rungs | Deployment robot/frame |
| --- | --- | --- | --- |
| `piper_fold` | `piper_fold_cloth_in_place` | `s0`, `s2`, `s2_pair` | Piper / Piper flange |
| `piper_x_native` | Seven Piper-X tasks | `s0`, `s2`, `s2_pair` | Piper-X / native Piper-X flange |
| `piper_x_aligned` | Same seven tasks, poses aligned during preparation | `s2`, `s2_pair` | Piper / Piper flange |
| `piper_all_aligned` | Piper folding plus seven aligned Piper-X tasks | `s2`, `s2_pair` | Piper / Piper flange |

The seven tasks are `piper_x_hang_mugs`, `piper_x_push_t`,
`piper_x_stack_blocks`, `piper_x_stack_bowls`, `piper_x_swap_t`,
`piper_x_sweep_blocks`, and `piper_x_sweep_the_trash`. The other tasks under
`latest` are not silently included. Each recorded English task prompt is
preserved. The implementation is synchronous; **training-time RTC, inference
RTC, AWBC, EMA and DAR are not enabled** in these VLAct experiments.

| Rung | State | Action | Physical command |
| --- | --- | --- | --- |
| `s0` | 14: left six joints + opening, right six joints + opening | 14: future joint minus anchor joint; absolute opening | Absolute joints |
| `s2` | 14: own-base xyz, continuous rotation vector, opening per arm | 20: body translation, first two rotation-matrix rows, absolute opening per arm | Absolute flange pose |
| `s2_pair` | 23: S2 + inter-arm xyz and two rotation rows | Same 20 values as S2 | Absolute flange pose |

Every chunk predicts **t+1 through t+50** at 30 Hz, relative to the same
observation at t. There are no fabricated tails and no episode crossing.
The final 50 frames cannot serve as anchors. Joint angles are radians;
positions and full gripper openings are metres. Grippers occur at state/S0
action indices 6 and 13, or S2 action indices 9 and 19. S0 is a delta-joint
baseline, not a recorded absolute-action baseline.

S2/S2-pair actions encode `inverse(T_t) @ T_future`. Rotation6d uses **rows**,
not columns. Their actions are identical; only pair conditioning changes.
The pair is `inverse(T_left) @ RIGHT_BASE @ T_right`, with right base at
`[0, -0.60, 0]` m and identity orientation. This is a nominal setup assumption,
not a measured calibration. Both use zero flange-to-TCP offset.

Piper poses are recomputed with the reference Piper MDH chain. Piper-X poses
come from recorded `observation.qpos_ee`, laid out as
`[xyz, quaternion_xyzw, opening]` per arm. Piper FK is never applied to
Piper-X joint trajectories. Labels use future follower observations, not
`action`, `action_mas`, or `action_ee`. Piper preparation records discrepancies
between recorded action and next follower joints. Native X preparation checks
agreement between the joint and EEF gripper channels.

## Read-only data and cache preparation

Run from the VLAct checkout root. The existing `.venv` is used by default;
set `PYTHON=/path/to/python` on another host. Runtime uses local model assets,
Torch/Accelerate, Transformers, NumPy/SciPy, PyArrow, PyAV, Pillow, OmegaConf,
Safetensors, MessagePack and WebSockets. There is no runtime dependency on
kai0, JAX, LeRobot, a robot SDK, or a remote Hugging Face service for training.
Robot SDK/ROS dependencies are separate from the training environment.

Pin the source snapshot for all comparisons:

```bash
export PIPER_SOURCE_ROOT="$(readlink -f /mnt/xiaoyu_teleop_data/piper/latest)"
# On 2026-09-22 this is /mnt/xiaoyu_teleop_data/piper/20260920.

bash examples/Piper/train.sh prepare piper_fold s2_pair
bash examples/Piper/train.sh prepare piper_x_native s2_pair
bash examples/Piper/train.sh prepare piper_x_aligned s2_pair
```

Caches live under `playground/Piper/registered/datasets/<robot_type>/<dataset_name>`,
with mixture statistics in `registered/mixtures/<data_mix>`; only
small numeric arrays and metadata are written. Videos and parquet are read
from the source. Video reads use resolved manifest paths, so a later change
to the `latest` symlink or source environment variables cannot redirect a run.
On a different host, prepare a new cache if the source mount path changes.
Never set `PIPER_CACHE_ROOT` inside the source tree; preparation rejects overlap.

The deterministic split uses seed 0 and 5% held-out episodes per task. Short
episodes (`length <= 50`) are excluded **after** splitting, without moving
an episode between partitions. `full_split`, exclusions, retained split and
source metadata fingerprints are saved. Piper-X training balances the seven
tasks; within each task it approximates uniform sampling over eligible
anchors. All representations share splits and statistics anchors.

The common-quantile sample budget is 8,192 times the number of registered
tasks, drawn from the weighted training distribution, with all 50 future targets
for each anchor. Actual per-task sample counts are recorded in the manifest;
they need not equal exactly 8,192. Held-out episodes never contribute.
Each task is registered separately in `DATASET_NAMED_MIXTURES`; the selected
`data_mix` controls sampling weights. Separate common statistics are saved per rung. These are sampled estimates, not exact
population quantiles. Existing caches reject source/settings changes; choose
a new `PIPER_CACHE_ROOT` when changing snapshot, alignment, split or sample count.

State is quantile-normalized, clipped to [-1,1], then discretized into 256
text bins by QwenPI. Actions use the same affine quantile normalization
without clipping. Constant dimensions map to zero with unit scale, preserving
invertibility. No binary gripper threshold, +/-pi clipping, joint-angle wrap
loss, or padded action coordinates are applied. Loss averages the actual 14
or 20 output dimensions; losses across representations are not a common
physical metric.

## Model initialization and training controls

**Full training must load `VLAct_Qwen3_Pretrain`.** The release is a hybrid
continued-pretraining checkpoint. The Piper loader selects and strictly loads
all `qwen_vl_interface` parameters, including a validated mapping for older
Qwen3 nesting. Missing, duplicate, unexpected or wrong-size backbone tensors
fail the run. The released task/action heads are excluded. QwenPI initializes
its action expert from local Qwen3-0.6B and initializes its PI adapters and
14/20-dimensional input/output projections for the new representation.

Default release path:

```bash
export PRETRAINED_CKPT="$PWD/playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt"
```

Set that variable to the actual completed `VLAct_Qwen3_Pretrain` download.
A `.incomplete` download is not accepted by the default path. The launcher's
`BASE_VLM` must contain compatible config/tokenizer and base weight files
(default local `Qwen3-VL-4B-Instruct-Action`); it does not replace the continued
pretraining checkpoint. `ACTION_EXPERT` overrides the local Qwen3-0.6B path.
`initialization.json` records what was loaded; pretraining normalization is
not used for Piper data.

All groups use seed 42, three ordered RGB views (head, left wrist, right wrist),
224x224 bilinear resize, frozen vision backbone, trainable language and action
weights, no image augmentation, BF16, gradient checkpointing, AdamW
(betas 0.9/0.95, weight decay 0), gradient norm clipping at 1, 60,000 updates,
1,000-step warmup to 5e-5, cosine decay to 5e-6, 10 inference flow steps,
and one sampled diffusion time per training example. Parameters and AdamW
states use BF16 in this native DDP implementation; it does not use DeepSpeed
FP32 master weights or optimizer sharding. This fixes controls
within VLAct; several controls differ from the kai0 pi05 experiment.

The inspected Piper folding videos decode as 640x480. All three views in the
inspected seven Piper-X tasks decode as **1280x720**, despite 640x480 metadata.
Training and serving use the same RGB validation and resize function, operating
on actual decoded shapes. This does not remove the camera/aspect-ratio domain
difference when a model trained on X videos is used on a Piper 640x480 cell.

## One-GPU smoke

```bash
# Smoke uses the completed VLAct_Qwen3_Pretrain release when available.
RUN_NAME=piper_s0_smoke bash examples/Piper/train.sh smoke piper_fold s0
RUN_NAME=piper_s2_smoke bash examples/Piper/train.sh smoke piper_fold s2
RUN_NAME=piper_pair_smoke bash examples/Piper/train.sh smoke piper_fold s2_pair
RUN_NAME=piper_x_pair_smoke bash examples/Piper/train.sh smoke piper_x_native s2_pair
RUN_NAME=aligned_pair_smoke bash examples/Piper/train.sh smoke piper_x_aligned s2_pair

# Match the full run's per-device batch and accumulation on one GPU.
BATCH_SIZE=2 ACCUMULATION=8 RUN_NAME=registered_pair_smoke \
  bash examples/Piper/train.sh smoke piper_all_aligned s2_pair
```

Smoke defaults: one GPU, per-device batch 1, accumulation 1, two updates,
two training and one held-out episode per task, 64 statistics anchors per task,
and a separate `registered_smoke` cache root. The default smoke checkpoint saves model and
metadata only; `SAVE_OPTIMIZER=1` also saves optimizer/scheduler/RNG state.
Smoke caches are explicitly rejected for full training, and the client refuses
hardware execution with a smoke checkpoint. A smoke loss is pipeline evidence,
not convergence or task-success evidence.

## Eight-GPU full training commands

Run the desired commands **sequentially** on a node with eight visible GPUs.
The launcher uses PyTorch DDP (`torchrun`), not the kai0 JAX/FSDP launcher.
Each GPU has batch 2 with 8 accumulation microbatches: global batch **128**.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PIPER_SOURCE_ROOT=/mnt/xiaoyu_teleop_data/piper/20260920
export PRETRAINED_CKPT="$PWD/playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt"

RUN_NAME=vlact_piper_s0_v1       bash examples/Piper/train.sh train piper_fold s0
RUN_NAME=vlact_piper_s2_v1       bash examples/Piper/train.sh train piper_fold s2
RUN_NAME=vlact_piper_s2_pair_v1  bash examples/Piper/train.sh train piper_fold s2_pair

RUN_NAME=vlact_piper_x_s0_v1      bash examples/Piper/train.sh train piper_x_native s0
RUN_NAME=vlact_piper_x_s2_v1      bash examples/Piper/train.sh train piper_x_native s2
RUN_NAME=vlact_piper_x_s2_pair_v1 bash examples/Piper/train.sh train piper_x_native s2_pair

RUN_NAME=vlact_aligned_s2_v1      bash examples/Piper/train.sh train piper_x_aligned s2
RUN_NAME=vlact_aligned_s2_pair_v1 bash examples/Piper/train.sh train piper_x_aligned s2_pair

RUN_NAME=vlact_all_s2_v1      bash examples/Piper/train.sh train piper_all_aligned s2
RUN_NAME=vlact_all_s2_pair_v1 bash examples/Piper/train.sh train piper_all_aligned s2_pair
```

`DRY_RUN=1` prints both preparation and training commands without dataset/GPU
access, including when the pretraining download is incomplete. Additional
variables: `GPUS`, `STEPS`, `BATCH_SIZE`, `ACCUMULATION`, `NUM_WORKERS`,
`PREPARE_WORKERS`, `STATS_SAMPLES`, `SAVE_EVERY`, `LOG_EVERY`, `OUTPUT_ROOT`,
`LOG_ROOT`, `PIPER_CACHE_ROOT`, `DATA_MIX`, `BALANCE_DATASET_WEIGHTS`. Keep matching controls across compared experiments.

Training writes `results/Piper/<RUN_NAME>/metrics.jsonl` and checkpoints at
`step_010000`, ..., `step_060000`. Each complete checkpoint includes model,
optimizer, scheduler, RNG state, next data position, resolved config, original
manifest, representation contract, initialization record and normalization
statistics. Writes use an `.incomplete` directory, published only after all
assets are present. Serving/resuming refuses unfinished checkpoints.

Resume using the same settings and run name:

```bash
RESUME="$PWD/results/Piper/vlact_piper_s2_pair_v1/step_030000" \
  RUN_NAME=vlact_piper_s2_pair_v1 \
  bash examples/Piper/train.sh train piper_fold s2_pair
```

Use the latest complete step when resuming in the same directory. Existing
checkpoint directories are never overwritten. A new `RUN_NAME` permits
branching from an older checkpoint. Changing world size, batch, schedule,
cache, representation or normalization is rejected on resume. Weights-only
smoke exports cannot resume training.

## Evaluation

Serve a complete checkpoint using [the deployment instructions](piper_deployment.md),
then run:

```bash
.venv/bin/python -m starVLA.piper.evaluate \
  --cache playground/Piper/registered/mixtures/piper_x_aligned --rung s2_pair \
  --host 127.0.0.1 --port 8000 --samples-per-task 32 \
  --output results/Piper/aligned_pair_held_out.json
```

The evaluator samples held-out anchors and reports per-task errors at future
steps 10/25/50 and over the whole chunk, alongside a stationary baseline.
S0 reports mean joint error in degrees and opening error in mm, without using
Piper FK on Piper-X. EEF groups report translation mm, SO(3) geodesic degrees,
and opening mm, left then right. Cache/coordinate fingerprints must match
the server. These are offline metrics, not real-robot task success.
