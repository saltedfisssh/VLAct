# Registered multi-dataset Piper training

Piper experiments use the existing
[`DATASET_NAMED_MIXTURES`](../starVLA/dataloader/gr00t_lerobot/mixtures.py)
registry. Each entry has the standard VLAct form:

```python
(dataset_name, sampling_weight, robot_type)
```

Each dataset has its own weight. Select a named combination with `data_mix`;
there is no separate cache-group weight configuration.

## Registered combinations

| `data_mix` | Datasets | Supported representations |
| --- | --- | --- |
| `piper_fold` | Piper cloth folding | S0, S2, S2-pair |
| `piper_x_native` | Seven native Piper-X tasks | S0, S2, S2-pair |
| `piper_x_aligned` | Same seven tasks, aligned to Piper flange | S2, S2-pair |
| `piper_all_aligned` | Piper folding plus the seven aligned Piper-X tasks | S2, S2-pair |

Every task is registered with weight `1.0` by default. Change an entry's
weight in `mixtures.py` or register another named combination, just as for the
existing Robotwin/InternA1 datasets. For example:

```python
DATASET_NAMED_MIXTURES["piper_my_tasks"] = [
    ("piper_fold_cloth_in_place", 1.0, "piper"),
    ("piper_fold_cloth_from_basket_640x480", 1.0, "piper"),
    ("piper_x_stack_bowls", 1.0, "piper_x_aligned"),
]
```

The `robot_type` selects the data transform:

- `piper`: recorded follower joints; Piper FK for EEF representations.
- `piper_x`: recorded native joints or quaternion EEF poses.
- `piper_x_aligned`: recorded X EEF poses converted to Piper flange during preparation.

Dataset paths are resolved below `data_root_dir`. No dataset is included by
an unrestricted directory scan. Unrecognized robot types, duplicate entries,
nonpositive weights and incompatible coordinate frames fail preparation.

The adapter is [`starVLA/dataloader/piper_datasets.py`](../starVLA/dataloader/piper_datasets.py).
It uses the normal `datasets.vla_data` fields:

```yaml
datasets:
  vla_data:
    dataset_py: piper_datasets
    data_mix: piper_all_aligned
    balance_dataset_weights: false
    representation: s2_pair
    image_size: [224, 224]
```

`cache_dir` is filled by the launcher and stored in the resolved run config.
The dedicated Piper trainer preserves the audited future-observation labels,
held-out partitions and deployment contract while using this registered loader.

## Weight semantics

`balance_dataset_weights` follows the existing codebase:

- `false` (default): use registered sampling weights directly, then normalize.
- `true`: multiply each registered weight by the number of eligible anchors,
  then normalize.

Within a task, sample eligible anchors uniformly. With equal registered
weights and the default setting, tasks receive equal sampling probability
regardless of dataset length. The resolved weights and actual probabilities
are recorded in the prepared manifest. Normalization statistics are estimated
from the same training distribution, with shared anchors across representations.

## Prepare and train

One launcher handles single- and multi-dataset experiments:

```bash
export PIPER_SOURCE_ROOT=/mnt/xiaoyu_teleop_data/piper/20260920
export PRETRAINED_CKPT="$PWD/playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt"

# Prepare the registered tasks and common training-only statistics.
bash examples/Piper/train.sh prepare piper_all_aligned s2_pair

# One GPU, two updates, isolated smoke cache.
RUN_NAME=registered_pair_smoke \
  bash examples/Piper/train.sh smoke piper_all_aligned s2_pair

# Eight GPUs: per-device batch 2, accumulation 8, global batch 128.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RUN_NAME=vlact_all_s2_v1 \
  bash examples/Piper/train.sh train piper_all_aligned s2
RUN_NAME=vlact_all_pair_v1 \
  bash examples/Piper/train.sh train piper_all_aligned s2_pair

# A custom registered combination; no new launcher or weight YAML is needed.
DATA_MIX=piper_my_tasks RUN_NAME=vlact_my_pair_v1 \
  bash examples/Piper/train.sh train
```

`DATA_MIX` overrides the positional mix name. `DRY_RUN=1` prints commands.
`BALANCE_DATASET_WEIGHTS=true` enables size-weighted sampling. Other controls
are described in the [training guide](piper_representation_ablation.md).

Numeric caches are prepared automatically under
`playground/Piper/registered/datasets/<robot_type>/<dataset_name>` and combined
statistics under `registered/mixtures/<data_mix>`. Smoke uses
`playground/Piper/registered_smoke`. `PIPER_CACHE_ROOT` selects another root.
Changing weights, source snapshot, calibration or statistics settings requires
a fresh cache root; stale cached statistics are rejected.

## Representation and validation constraints

S0 cannot mix Piper and Piper-X joints. Cross-robot S2/S2-pair uses aligned X
poses; native X and Piper frames cannot silently share a policy. State/action
widths, FPS, horizon, future offset, TCP and pair-base assumptions must agree.

Every task keeps its deterministic train/held-out episode split. Duplicate
source episodes and cross-partition overlaps are rejected. Mixed statistics
use only training samples; member-specific normalization is not applied before
mixing. Videos and parquet remain read-only. Real decoded RGB sizes are used,
including 1280x720 X videos whose metadata incorrectly advertises 640x480.

A registered mixed checkpoint contains common statistics and the resolved data
contract. Piper/aligned-X checkpoints use the plain Piper client; native-X
checkpoints use the native-X client. Held-out evaluation reports each registered
task separately. See [deployment](piper_deployment.md) and
[recorded validation](piper_validation.md).
