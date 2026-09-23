# Piper validation record — 2026-09-22

Validation ran on this server's single NVIDIA A800 80 GB, using the checkout's
`.venv` (Python 3.10, PyTorch 2.11.0+cu128, Transformers 4.57.0,
Accelerate 1.5.2). Source data was read from
`/mnt/xiaoyu_teleop_data/piper/20260920`, resolved from `latest`. All generated
caches, checkpoints and reports are under this VLAct checkout.

## Final registered-data training path

The final launcher uses `DATASET_NAMED_MIXTURES` and individual task weights.
All default registered weights are `1.0`; `balance_dataset_weights=false`.
These three runs used the actual completed release:

```text
playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt
```

Each run's `initialization.json` records the 11,736,465,942-byte source file
and successful strict loading of 714 Qwen-VL backbone tensors. The original
release's action heads are excluded; the action expert uses Qwen3-0.6B with
new PI adapters/projections, as described in the [training guide](piper_representation_ablation.md).

| Run under `results/Piper/` | Registered mix | Rung | Batch / accumulation | Update 1 loss | Update 2 loss | Peak allocated GB |
| --- | --- | --- | --- | --- | --- | --- |
| `registered_pretrained_s0_smoke_20260922` | `piper_x_native` | S0 | 1 / 1 | 7.467332 | 6.216426 | 51.085 |
| `registered_pretrained_s2_smoke_20260922` | `piper_all_aligned` | S2 | 1 / 1 | 6.308983 | 7.049491 | 51.073 |
| `registered_pretrained_pair_smoke_20260922` | `piper_all_aligned` | S2-pair | 2 / 8 | 6.321864 | 6.634979 | 51.087 |

All completed forward/backward/optimizer updates and published complete
`step_000002` checkpoints with weights, normalization, data contract, source
manifest and initialization metadata. These final smoke exports omit optimizer
state. The pair run checks the intended eight-GPU run's per-device batch and
accumulation; its one-GPU global batch is 16, while eight GPUs use 128.

Reproduce with new run names:

```bash
RUN_NAME=registered_s0_check bash examples/Piper/train.sh smoke piper_x_native s0
RUN_NAME=registered_s2_check bash examples/Piper/train.sh smoke piper_all_aligned s2
BATCH_SIZE=2 ACCUMULATION=8 RUN_NAME=registered_pair_check \
  bash examples/Piper/train.sh smoke piper_all_aligned s2_pair
```

Before the registered-loader refactor and completed pretraining download,
two-update base-Qwen smoke runs also covered all eight individual experiment
cells: Piper S0/S2/S2-pair, native-X S0/S2/S2-pair and aligned-X S2/S2-pair.
Their logs/checkpoints are retained as `piper_*_smoke_v2_20260922` under
`results/Piper`; the table above is the validation of the final registration
and release-initialization path. An earlier Piper S0 smoke also saved a full
Accelerate optimizer/scheduler/RNG checkpoint; full interruption/resume and
multi-rank checkpoint publication have not been exercised end to end here.

## Full data preparation

All four full-cache combinations are prepared under
`playground/Piper/registered/mixtures/`: `piper_fold`, `piper_x_native`,
`piper_x_aligned` and `piper_all_aligned`. Member caches live under
`registered/datasets/<robot_type>/<dataset_name>`. The retained episode counts
are:

| Dataset | Training | Held out |
| --- | --- | --- |
| `piper_fold_cloth_in_place` | 1,191 | 63 |
| `piper_x_hang_mugs` | 194 | 10 |
| `piper_x_push_t` | 194 | 10 |
| `piper_x_stack_blocks` | 190 | 10 |
| `piper_x_stack_bowls` | 207 | 11 |
| `piper_x_swap_t` | 200 | 10 |
| `piper_x_sweep_blocks` | 190 | 10 |
| `piper_x_sweep_the_trash` | 473 | 25 |
| Total for `piper_all_aligned` | 2,839 | 149 |

The 15-frame Piper episode 298 is excluded after splitting because it cannot
provide 50 future observations. Full-cache statistics use training data only;
the eight-task common-statistics budget is 65,536 anchors. Source files and
videos are neither modified nor copied into these caches.

## Reload, inference and client checks

The final registered S2-pair checkpoint was reloaded into the GPU policy server.
Eight real held-out observations, one per registered task, were sent through
the local MessagePack/WebSocket client. All returned finite `(50, 20)` actions;
a malformed state was rejected. The full report, including contract and
physical errors, is
[`serve_registered_pair.json`](../results/Piper/validation/serve_registered_pair.json).
Measured model inference time was approximately 1.94 s for the first request
and 1.09 s for subsequent requests; these observations are not an end-to-end
robot latency benchmark.

Earlier reload/WebSocket probes covered Piper S0 and all seven aligned-X tasks:
[`serve_s0.json`](../results/Piper/validation/serve_s0.json),
[`serve_aligned_pair.json`](../results/Piper/validation/serve_aligned_pair.json).
Decoded-video inspection confirmed Piper 640x480 and all three views of each
inspected X task at 1280x720; training and serving now share the same resize
path and accept the actual RGB dimensions.

The regular `build_dataloader(..., dataset_py="piper_datasets")` path also
produced a batch of two with state `(1, 23)`, action `(50, 20)` and three
224x224 images per sample. The CPU client bundle was exported and its mock
S0/S2/S2-pair cycles run from the bundle directory, without importing the
training checkout. The tested archive is
[`client_bundle_final.tar.gz`](../results/Piper/client_bundle_final.tar.gz);
[`registered_summary.json`](../results/Piper/validation/registered_summary.json)
contains the machine-readable run summary. See [deployment](piper_deployment.md).

## Regression checks and remaining limits

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p test_piper.py -v`
passed **23 tests**. Coverage includes future/anchor geometry, rotation rows,
native/aligned frames, grippers, held-out isolation, pinned source paths,
registry weights/profile consistency, size weighting, mixture overlap
rejection, common normalization, pretrained key mapping and mixed RGB sizes.
Ruff on new Python modules, Python compilation, launcher shell syntax and
`git diff --check` also passed.

Two optimizer updates demonstrate a working pipeline, not convergence or task
success. No eight-GPU execution, 60,000-step training, ROS/CAN publication,
hardware IK/tracking or physical rollout was performed. Nominal frame/base
alignment still needs to match the actual cell. Training uses native DDP with
BF16 parameters and AdamW states; eight-GPU communication overhead and memory
have not been measured. [Full training commands](piper_representation_ablation.md)
and [registered mixtures](piper_mixed_training.md) are ready for the training node.
