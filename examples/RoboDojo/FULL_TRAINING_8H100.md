# 8×H100 全量原子动作 + VQA 联合训练

本轮在 `root@10.10.2.240:22` 的现有 VLAct `.venv` 中运行，tmux 会话为 `vlact-full`。当前机器与服务器共享磁盘，不需要复制数据或创建新环境。实际阶段以运行目录的 `status.json` 为准；尚未得到 30,000 步完成记录时，不能称为全量训练完成。

运行目录：

`/kpfs_ssd/data/wzy/VLAct/results/RoboDojo/qwen36_full_8h100_20260924`

## 数据与配方

- 35 个任务、3,500 条轨迹、1,859,602 帧；每个任务按完整轨迹分为 90 条训练、10 条验证，总计 3,150/350 条。动作与 VQA 使用同一划分，归一化统计仅来自训练集。
- 训练 VQA 123,825 对、验证 13,584 对，包含计数、属性、空间关系。框与关键点的坐标约定尚未核验，本轮排除。
- 使用 Qwen3-VL-4B-Instruct-Action 的架构和 tokenizer，严格载入 VLAct_Qwen3_Pretrain 第 100,000 步的主干权重，新建 14 维动作头；完整微调主干与动作头。
- 动作条件：整任务目标 + 当前原子任务 + 当前三相机 RGB。动作窗口 50，跨原子任务/轨迹边界的动作目标屏蔽。状态向量不作为模型输入。
- 损失：动作 L1 + 0.1 × VQA 答案 token 的交叉熵。VQA 的问题与图像提示 token 不参与监督。
- 每卡动作 batch 8，累积 2 次，8 卡全局动作 batch 128；每卡每次 VQA batch 1。
- BF16、ZeRO-2、梯度检查点，不使用 CPU 优化器卸载。
- 正式训练 30,000 步，warmup 1,000 步，主干学习率 1e-5、动作头 1e-4；每 500 步固定抽样验证，每 500 步保存权重及优化器/RNG/数据进度。
- 验证抽样覆盖 35 个任务；这些离线指标不等于仿真成功率。

标注为自动伪标签，`annotations_reviewed=false`。本次明确授权全量使用，并未将其标为人工逐条验收。第 1710 条原来缺失计划，已独立重标并根据真实动作起点校正一次抓取边界；保留原始分片、模型输出和审计记录，原 VQA 未改变。服务器端全量复验通过，错误和警告均为 0。修复记录：

`/kpfs_ssd/data/wzy/lerobot/outputs/robodojo/robodojo_annotated_qwen36/repair_1710.json`

## 查看与恢复

查看后台日志：

```bash
ssh -t -p22 root@10.10.2.240 'tmux attach -t vlact-full'
```

本次配方的完整可执行脚本：

`/kpfs_ssd/data/wzy/VLAct/results/RoboDojo/qwen36_full_8h100_20260924/train_full.sh`

该脚本与后台正式训练使用相同配置。当前任务已由后台调度，无需再启动第二次。

中断后，在原 H100 服务器上选择一个已包含 `progress.json` 的训练状态目录恢复；例如第 1,000 步：

```bash
cd /kpfs_ssd/data/wzy/VLAct
bash results/RoboDojo/qwen36_full_8h100_20260924/train_full.sh \
  --resume results/RoboDojo/qwen36_full_8h100_20260924/full_train/training_states/step_1000
```

仅有 `checkpoints/steps_1000_pytorch_model.pt` 是推理权重，不是完整续训状态。恢复时保持数据、8 卡数量和训练配方一致。

## 实际验证证据

运行目录保存：

- `annotation_gate.json`、`cache_gate.json`：完整标注与每任务 90/10 划分。
- `smoke_gate.json`：8 卡、全局 batch 128 的真实前 4 步更新，主干、动作头和语言词嵌入探针均有变化。
- `verify_native_smoke.json`、`verify_robodojo_smoke.json`：严格加载权重后，三相机输入产生有限的 [1, 50, 14] 动作；验证 VQA 答案损失可计算。
- `resume_gate.json`：从第 4 步恢复优化器和 RNG 后继续更新至第 8 步；文件存在且通过才会启动正式训练。
- `full_train/config.full.yaml`、`training_inputs.jsonl`、`vqa_training_inputs.jsonl`、`metrics.jsonl`：正式配方、实际输入审计及训练/验证指标。
- `full_training_gate.json`、`verify_full_checkpoint.json`：30,000 步完成后才由后台生成；`status.json` 最终变为 `completed`。

正式训练从预训练主干与新动作头重新开始，不继承 smoke 的更新。后台出错会记录 `failed` 和异常，不会把失败当成训练完成。

## 正确评测

参见 [ORACLE_EVALUATION.md](ORACLE_EVALUATION.md)。本次训练不会自动得到子任务规划器；VQA 联合训练也不能替代规划与子任务切换。在线子任务 oracle 应从当前仿真状态决定当前指令，并在切换时清空对应环境的旧动作缓存，不能按专家轨迹的时间戳推进。

当前尚无已验证的 35 任务在线 oracle provider，未运行真实仿真 rollout。现有仿真注册表覆盖数据中的 34 个任务，DLC 拼 RoboDojo 任务缺少实现/配置；不得把 34 任务结果称为完整 35 任务成绩。

H100 可用于训练与策略推理；当前 Isaac Sim 5.1 的 RGB 相机仿真应放在有 RT cores 的 RTX 节点，通过 RoboDojo 的 server/client 接口连接。见 [NVIDIA 官方硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。

2026-09-25：解码故障已修复，并从第 1,000 步完整状态直接续训。详见 [恢复记录](RECOVERY_20260925.md)。
