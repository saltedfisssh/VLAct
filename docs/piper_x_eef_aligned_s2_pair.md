# Piper-X to Piper aligned S2 / S2-pair in VLAct

This follows the conventions in the same-named kai0 document. The implementation
is self-contained in `starVLA/piper`; it does not import the reference checkout.
See [all experiment commands](piper_representation_ablation.md),
[mixed training](piper_mixed_training.md), and [deployment](piper_deployment.md).

The native Piper-X recordings use their own zero-offset flange convention.
Piper deployment obtains its flange from Piper FK. The reference nominal
mapping is `T_X = A @ T_P @ C`, where `A` is identity and `C` is local
`Rz(-90 degrees)` with zero translation. Preparation applies:

```text
T_P = inverse(A) @ T_X @ inverse(C)
```

Thus nominal alignment keeps xyz and opening unchanged and right-multiplies
recorded rotation by `Rz(+90 degrees)`. It is an inherited mounting-axis
assumption, not a calibration measured on this server. A measured calibration
may include per-arm base and TCP translation. Openings remain metres, without
rescaling or clipping. Pair geometry uses parallel, forward-facing bases with
60 cm spacing for both source and target robots.

## Build and train

```bash
export PIPER_SOURCE_ROOT=/mnt/xiaoyu_teleop_data/piper/20260920
bash examples/Piper/train.sh prepare piper_x_aligned s2_pair

# Separate smoke cache; uses the completed VLAct_Qwen3_Pretrain release.
RUN_NAME=aligned_pair_smoke bash examples/Piper/train.sh smoke piper_x_aligned s2_pair

# Full training uses the completed VLAct_Qwen3_Pretrain download.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PRETRAINED_CKPT="$PWD/playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt"
RUN_NAME=vlact_aligned_s2_v1 bash examples/Piper/train.sh train piper_x_aligned s2
RUN_NAME=vlact_aligned_pair_v1 bash examples/Piper/train.sh train piper_x_aligned s2_pair
```

Both rungs share `playground/Piper/registered/mixtures/piper_x_aligned`, the seven-task split,
future observations and statistics anchors. State width is 14 for S2, 23 for
S2-pair; both have the same 20-dimensional body-relative action. Their S2-pair
block is `inverse(T_left) @ RIGHT_BASE @ T_right`, represented by xyz and the
first two rotation rows. The nominal per-arm mapping is recorded in settings,
manifest and checkpoint; native and aligned caches are distinct.

Use a measured JSON instead with `PIPER_ALIGNMENT=/absolute/path/calibration.json`
and a new `PIPER_CACHE_ROOT`. The format is:

```json
{
  "version": 1,
  "convention": "T_X = A @ T_P @ C",
  "calibration_id": "installed_cell_measurement_v1",
  "nominal": false,
  "left": {"A": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "C": [[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]},
  "right": {"A": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "C": [[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]}
}
```

The numeric matrices above merely illustrate the nominal axes. Replace them
with actual measured values before identifying the file as a measured
calibration. The loader validates proper SE(3), finite values and exact schema.

## Deploy aligned policies

Serve the matching complete checkpoint on the GPU host:

```bash
.venv/bin/python -m starVLA.piper.serve \
  --checkpoint results/Piper/vlact_aligned_pair_v1/step_060000 --port 8000
```

The robot uses the plain Piper client; **no additional Piper-X transfer is
applied**:

```bash
python -m starVLA.piper.run --robot piper --rung s2_pair \
  --backend ros --policy websocket --host GPU_HOST \
  --prompt 'Stack the three bowls together.' \
  --ros-topics ros_topics.json --cycles 1
```

Without `--execute`, this reads observations and logs decoded predictions.
The server normalizes state and unnormalizes actions once. The client decodes
body actions using the immutable observation anchor. Native Piper-X clients
reject aligned policies, and Piper clients reject native Piper-X policies.
These VLAct commands intentionally expose no client-side transfer switch,
which avoids double-applying alignment.

S0 cannot use the alignment conversion: a pose-frame mapping does not define
a Piper joint trajectory from Piper-X joint angles. Use the `piper_x_native s0`
experiment only on native Piper-X, or `piper_fold s0` on Piper.

Tool-axis alignment does not align camera extrinsics, view framing, gripper
geometry or scene distribution. The source X opening can exceed the Piper
client's 0.08 m default; the client records clipping. Confirm TCP, base spacing,
opening calibration and tracking on the actual cell before interpreting a
rollout. No physical rollout or transfer-success result is claimed here.
