# RoboDojo VLAct / QwenOFT 训练

使用本仓库现有 `.venv`，所有命令通过 `uv run --no-sync` 执行。无需新建环境。
训练入口使用本仓库原始 `QwenOFT` 模型、L1 动作损失及参数分组，独立的数据适配和训练循环位于本目录。
RoboDojo 与其 XPolicyLab 子模块不需要修改。

原子任务与打标 VQA 联合训练的实测结果见 [SUBTASK_SMOKE_RESULT.md](SUBTASK_SMOKE_RESULT.md)；先前任务指令基线见 [SMOKE_RESULT.md](SMOKE_RESULT.md)。

正式 8×H100 的等待、训练与恢复入口为 `after_annotation_train.py`。本次明确授权使用全量自动伪标签，因此数据准备增加 `--allow-unreviewed-training`，仍保留 `annotations_reviewed=false`；这与仅允许最多 10 步的 `--allow-unreviewed-smoke` 区分。Oracle 评测、离线验证入口和本次运行状态见 [ORACLE_EVALUATION.md](ORACLE_EVALUATION.md)。

## 模型与数据约定

- 架构和 tokenizer：本地 `Qwen3-VL-4B-Instruct-Action`。
- 初始化：`VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt`，严格加载 `qwen_vl_interface` 骨干；新建 14D OFT 动作头，训练步数从 0 开始，全参数更新。
- 不使用已经在 RoboDojo 上训练的发布模型作为初始化，避免污染这次数据划分的验证结果。
- 输入：头部、左腕、右腕三个 RGB 视角，224×224，使用与 RoboDojo policy 一致的 RGB / INTER_AREA 缩放；原始任务指令。`include_state=false`，与原始 VLAct OFT 的输入一致；不同于带离散状态提示的已发布 RoboDojo 配方。
- 动作：50 步、14 维绝对关节值，保持 `[左臂6, 左夹爪1, 右臂6, 右夹爪1]` 原始顺序。连续夹爪不二值化。
- 所有维度按训练集 q01/q99 归一化、裁剪到 [-1,1]；零范围维度沿用上游转换规则。
- episode 末尾及可选子任务边界外的目标被屏蔽，动作 chunk 不跨轨迹。
- 源 LeRobot v3 数据保持只读。缓存只包含 action/state/timestamp 数组、视频位置、划分清单和统计；AV1 视频按原始时间偏移读取，不复制或改写。

## 单卡 smoke

在 VLAct 仓库根目录执行：

```bash
bash examples/RoboDojo/smoke.sh
```

默认从两个任务各选两条轨迹，其中每个任务一条训练、一条验证；batch=1、梯度累积=2、2 次优化步骤。
单卡 80GB 放不下全参数 AdamW 状态，smoke 使用 ZeRO-2 + 优化器 CPU 卸载，需要约 100GB 以上可用主机内存；模型参数均可训练。
完整训练的八卡配置不启用 CPU 卸载。

```bash
uv run --no-sync python -m unittest examples.RoboDojo.test_data -v
uv run --no-sync python -m examples.RoboDojo.verify_checkpoint \
  --checkpoint results/RoboDojo/<run_id>/checkpoints/steps_2_pytorch_model.pt \
  --cache results/RoboDojo/data_smoke
```

检查其他服务器上的路径时，可向验证脚本传入 `--base-vlm` 和 `--source-override`。
`verify_checkpoint.py --policy-source /path/to/RoboDojo/XPolicyLab/policy/starVLA/source_starvla` 改用预置 policy 的模型代码进行严格加载与推理。
此验证不启动仿真器，不代表任务成功率。正式仿真应从 RoboDojo 的 `scripts/robodojo.sh` 或预置 starVLA policy 入口启动其服务端，以满足 `runtime_contract` 握手；上游 VLAct 的通用 websocket 服务端不能替代该入口。

## 单机 8×H100 训练

```bash
cd /kpfs_ssd/data/wzy/VLAct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_ID=vlact_robodojo_oft_8h100 \
bash examples/RoboDojo/train_8xh100.sh
```

其他服务器通过环境变量指定本地路径：

```bash
SOURCE=/path/to/RoboDojo_lerobot_v30_video \
BASE_VLM=/path/to/Qwen3-VL-4B-Instruct-Action \
PRETRAINED_CKPT=/path/to/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt \
RUN_ROOT=/path/to/training_outputs \
RUN_ID=vlact_robodojo_oft_8h100 \
bash examples/RoboDojo/train_8xh100.sh
```

默认设置：BF16、FlashAttention 2、梯度检查点、ZeRO-2、每卡 batch 8、累积 2，global batch 128；骨干学习率 1e-5、动作头 1e-4，warmup 1000，训练 30000 步。
这是起始训练预算，需根据验证 loss 和仿真成功率调整；尚未在 8 卡机器上实测吞吐和显存。
`BATCH_SIZE=4 GRAD_ACCUM=4` 可保持 global batch 128 并减小每卡微批大小。
`MAX_STEPS` 可调整总步数，后接 `trainer.*=value` 可覆盖配置。

数据按 seed=42 在每个任务内随机划分 90/10：35 个任务 × 90=3150 条训练，35×10=350 条验证；归一化统计仅使用训练部分。
验证轨迹只用于离线动作误差，不等同于仿真测试集。
日志中的 `val_l1` 使用固定、按任务均衡的验证子集，默认每卡最多 8 个 batch；8 卡正式配置覆盖全部 35 个任务。`validation_sampling.json` 记录精确样本索引与任务覆盖。它仍是快速诊断；正式比较应另外运行完整离线验证及相同配置的 RoboDojo 仿真测试。

预览命令不加载 GPU、不写训练结果：

```bash
DRY_RUN=1 bash examples/RoboDojo/train_8xh100.sh
```

每个 run 包含：

- `config.yaml` / `config.full.yaml`、`dataset_statistics.json`：RoboDojo policy 所需配置和 `arx_x5` 反归一化统计。
- `data_manifest.json`：源数据 SHA256、每条源 episode 的训练/验证归属、相机视频偏移。
- `training_setup.json`、`metrics.jsonl`：参数量、world size、有效 batch、训练/验证 loss。
- `checkpoints/steps_<N>_pytorch_model.pt`：可独立推理的完整权重。
- `training_states/step_<N>/`：包含优化器、调度器和 RNG 的恢复状态（smoke 默认不保存）。
- `parameter_updates.json`、`completed.json`：训练正常完成以及骨干/动作头确实发生更新的检查。

每 5000 步以及最终保存完整权重和恢复状态，需预留足够磁盘空间。保持相同 world size、batch、数据缓存和训练配置，从已有状态继续：

```bash
PREPARE=0 RUN_ID=<同一个run_id> \
bash examples/RoboDojo/train_8xh100.sh \
  --resume results/RoboDojo/<run_id>/training_states/step_5000
```

不要将 `--resume` 指向单独 `.pt` 权重；它不包含优化器状态。

## 原子任务标注如何使用

`smoke.sh` 验证任务指令基线；原子任务条件训练使用下面的独立入口：

```bash
bash examples/RoboDojo/smoke_subtasks.sh
```

它直接使用当前 `lerobot/outputs/robodojo/review_35_qwen3vl30b` 中的 35 条自动标注，输入为“任务目标 + 当前原子指令 + 三路图像”，监督目标为当前原子任务区间内的动作。
现有自动标注尚有语义错误，缓存明确记录 `annotations_reviewed=false, smoke_only=true`；此诊断缓存最多允许 10 个优化步骤，默认运行 4 步，不把它标成已验收数据。
每个任务只有一条标注轨迹，因此按整个任务留出 4 条验证轨迹，31 条训练；该验证只检查流程，不能作为同任务泛化成绩。
`training_inputs.jsonl` 记录真实送入骨干模型的 tokenized prompt、原子指令、轨迹/帧编号与动作掩码，能够检查原子文本确实参与训练。
`verify_checkpoint.py` 同样检查重载推理时的真实输入。原始 3500 条轨迹编号保存在 `original_episode_index`，标注数据的 0–34 编号另存为 `source_episode_index`。

若要复现或换路径，可设置 `SOURCE`、`CACHE_DIR`、`RUN_ID` 和 `MAX_STEPS`；`DRY_RUN=1` 只打印命令。

若人工验收后要训练“任务指令 + 当前原子任务 + 图像 → 动作”，可建立独立缓存：

```bash
uv run --no-sync python -m examples.RoboDojo.data \
  --source /path/to/reviewed_annotated_lerobot_v3 \
  --output results/RoboDojo/data_reviewed_subtasks \
  --language-mode subtask --annotations-reviewed
CACHE_DIR="$PWD/results/RoboDojo/data_reviewed_subtasks" \
RUN_ID=vlact_robodojo_subtask_8h100 \
bash examples/RoboDojo/train_subtasks_8xh100.sh
```

正式数据每个任务至少两条轨迹时，默认在每个任务内划分训练/验证。仍只有一条时，准备命令需加 `--split-unit task`，按整个任务留出，不能随机切同一轨迹的帧。
8 卡入口从缓存清单读取源数据路径；换服务器时用 `SOURCE=/new/path` 覆盖。它检查缓存确实为 `subtask` 模式，防止误用纯任务指令缓存。

该模式只读取 `language_persistent` 中 `style=subtask` 且时间戳不晚于当前帧的最新文本；不把未来 plan、memory 或 VQA 答案注入输入。
当前原子任务的开始时间改变时（包括两段文本恰好相同），后续动作标签被屏蔽，避免以旧指令监督新子任务动作。
这是 **以子任务文本为条件的动作训练**，没有额外训练模型生成子任务文本。
实际仿真必须由在线规划器或用户提供相同格式的当前子任务；直接使用测试轨迹的真值子任务属于 oracle 条件，需单独报告，不能与端到端基线混比。
若要让同一模型同时生成子任务与动作，需要另加文本监督与在线切换机制，此训练入口未实现该目标。

## 将打标生成的 VQA 接入 co-training

```bash
bash examples/RoboDojo/smoke_subtasks_vqa.sh
```

此入口读取同一份标注数据的 `language_events`，按 **episode + frame + camera** 配对 `style=vqa` 的 user 问题和 assistant 答案。
Writer 会按 role 排序，所以不能假设问题与答案相邻；缺失或不唯一的配对会报错。
每个 VQA 只输入其对应相机在该帧的图像和问题；原子任务动作分支继续输入三路图像、任务目标和当前原子指令。

联合目标为 `L = L_action + 0.1 * L_vqa`。VQA 使用语言模型的答案 token 交叉熵，问题、图像位置、padding 均以 `-100` 屏蔽；每个微批的两项 loss 合在一次 backward 中，避免 DeepSpeed 将两次 backward 当成额外优化步骤。
`metrics.jsonl` 分别记录 `train_l1`、`vqa_ce`、`total_loss`、`val_l1`、`val_vqa_ce`；`vqa_training_inputs.jsonl` 记录相机、问题、答案、真实监督文本及被屏蔽的 prompt token 数。
VQA 答案在 VQA 分支作为 teacher-forcing 的 assistant 目标，不会被插入动作分支的输入。

当前 35 条轨迹共 519 对 VQA。本轮使用 count、attribute、spatial 三类的 303 对（训练 279、验证 24），划分与动作数据保持一致。
另有 bbox 90 对、keypoint 126 对：标注提示要求像素坐标，但现有答案中存在尺度不明或越界值，且训练图像会缩放，因此本轮排除并在缓存清单中记录；需要先校正标注时的图像尺寸和坐标语义再启用。
保留的 VQA 同样是未验收的伪标签，此 smoke 只证明联合训练流程，不证明标签正确或模型性能提高。

正式训练使用已验收数据准备缓存时，增加 `--include-vqa`：

```bash
uv run --no-sync python -m examples.RoboDojo.data \
  --source /path/to/reviewed_annotated_lerobot_v3 \
  --output results/RoboDojo/data_reviewed_subtasks_vqa \
  --language-mode subtask --annotations-reviewed --include-vqa
CACHE_DIR="$PWD/results/RoboDojo/data_reviewed_subtasks_vqa" \
bash examples/RoboDojo/train_subtasks_8xh100.sh \
  datasets.vqa_data.enabled=true datasets.vqa_data.loss_weight=0.1 \
  datasets.vqa_data.per_device_batch_size=1
```

checkpoint 保留普通 QwenOFT 的参数名称，可由原有 VLAct / RoboDojo 动作模型严格加载。`verify_checkpoint.py` 对联合训练权重额外检查一条验证 VQA 的答案交叉熵；这不是自由生成准确率评估。

本次 8×H100 全量原子动作 + VQA 联合训练的配方、运行证据与恢复方式见 [FULL_TRAINING_8H100.md](FULL_TRAINING_8H100.md)。
