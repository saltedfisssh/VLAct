# 原子任务与标注 VQA 联合训练 smoke

时间：2026-09-24 UTC。原子任务动作训练、原子任务 + 标注 VQA 联合训练均完成 4 次优化并保存权重。复用 VLAct 现有 uv 环境，没有新建环境或安装依赖。

## 数据与训练含义

- 源标注：`/kpfs_ssd/data/wzy/lerobot/outputs/robodojo/review_35_qwen3vl30b`，35 个任务各一条轨迹，共 19,468 帧、614 个原子片段。源数据保持只读。
- 按任务留出验证：31 条训练（17,694 帧）、4 条验证（1,774 帧），动作与 VQA 共用划分。只有一条轨迹的任务不能再进行任务内 episode 划分；此验证仅检查流程。
- 动作输入：三路 RGB 图像 + 任务目标 + 当前原子指令；目标为最多 50 步、14 维绝对关节动作。只读取当前时间已经开始的最新原子任务，跨原子片段或轨迹边界的动作不计入 loss。
- VQA 输入：标注指定帧、指定相机的单张图像 + 原始问题；目标为标注生成的 JSON 答案。问题、图像占位和 padding 不计算语言 loss，VQA 答案不进入动作输入。
- 共 519 对 VQA；使用 count 102、attribute 99、spatial 102，共 303 对（训练 279、验证 24）。bbox 90、keypoint 126 暂时排除：已有坐标尺度不明或越界问题，需先核对标注时的图像尺寸和坐标语义。
- 自动标注未人工验收，缓存保留 `annotations_reviewed=false, smoke_only=true`。这次没有训练模型生成原子任务文本，也没有证明伪标签正确。

## 实测结果

单张 NVIDIA A800-SXM4-80GB，BF16、ZeRO-2、梯度检查点、优化器 CPU 卸载。45.087 亿参数均设置为可训练；初始化为 VLAct 预训练骨干和新建 14D OFT 动作头。

| 项目 | 原子任务动作 | 原子任务动作 + VQA |
| --- | ---: | ---: |
| 优化步数 | 4 | 4 |
| 每卡动作 batch / 梯度累积 | 1 / 2 | 1 / 2 |
| 实际训练动作样本 | 8 | 8 |
| 实际训练 VQA 样本 | 0 | 8 |
| 最后一步动作 L1 | 2.689671 | 2.721783 |
| 最后一步 VQA CE | — | 1.490417 |
| 最后一步联合 loss | — | 2.870825 |
| 验证动作 L1（1 个 batch） | 2.420332 | 2.438997 |
| 验证 VQA CE（1 个 batch） | — | 2.952471 |
| 训练峰值已分配显存 | 25.70 GiB | 25.70 GiB |

联合目标为 `L_action + 0.1 * L_vqa`，每个微批只进行一次联合 backward。不同 batch 的四步 loss 不能用来判断收敛或比较模型优劣。

- 两次运行的实际 token 输入审计均确认 8/8 个动作样本包含当前原子指令。有效动作长度为 `[40, 50, 45, 44, 8, 50, 19, 35]`，其余位置被屏蔽。
- 联合训练记录 8 条训练 VQA 和 1 条验证 VQA；所有记录的 prompt 和 padding 均被屏蔽，答案 token 确实受监督。例如原始 episode 3200、帧 250、头部相机的问题为 “How many T-shaped blocks are on the table?”，目标为 `{"count": 2, "label": "T-shaped blocks"}`。这是伪标签示例，并未人工确认计数正确。
- 联合训练抽样的骨干参数有 20,559/65,536 个元素变化，动作头 65,536/65,536，绑定的语言输出/输入 embedding 13,857/65,536。
- 联合训练权重分别由 VLAct 原仓库和 RoboDojo 预置 starVLA 模型代码以 `strict=True` 成功重载；两边均用真实三相机输入预测有限的 `[1, 50, 14]` 动作，并确认当前原子指令进入模型输入。
- 重载后同一条验证 VQA 的 teacher-forcing 答案 CE：VLAct 2.945229，RoboDojo starVLA 2.928782；均为有限值。这不等于自由生成回答正确。
- 10 项测试通过：当前标签选择、同时间冲突、相同文本的新片段边界、任务留出、归一化与统计、真实 OFT loss 的边界梯度屏蔽、VQA 相机配对、坐标排除、真实 Qwen 模板的答案监督和 padding。
- 所有 19,468 帧完成边界审计；训练/验证缓存不跨任务共享轨迹。Python 编译、脚本语法和 8 卡命令 dry-run 通过。

## 权重与复现

联合训练权重：`results/RoboDojo/subtask_vqa_smoke_20260924/checkpoints/steps_4_pytorch_model.pt`。
原子任务动作权重：`results/RoboDojo/subtask_smoke_20260924/checkpoints/steps_4_pytorch_model.pt`。

完整机器可读证据：[联合训练 smoke_report.json](../../results/RoboDojo/subtask_vqa_smoke_20260924/smoke_report.json)。该运行目录中的 `validation/` 保存测试、训练、两种重载日志及全帧边界审计；`training_inputs.jsonl` 和 `vqa_training_inputs.jsonl` 保存实际模型输入。

在 VLAct 仓库根目录复现：

```bash
bash examples/RoboDojo/smoke_subtasks_vqa.sh
```

正式 8×H100 训练使用已验收的标注建立含 VQA 的缓存，再运行：

```bash
CACHE_DIR="$PWD/results/RoboDojo/data_reviewed_subtasks_vqa" \
bash examples/RoboDojo/train_subtasks_8xh100.sh \
  datasets.vqa_data.enabled=true datasets.vqa_data.loss_weight=0.1
```

缓存准备、其他服务器的路径覆盖和 checkpoint 重载命令见 [README.md](README.md)。当前未执行 8 卡训练、VQA 自由生成准确率评估或 RoboDojo 仿真 rollout。实际部署仍需在线提供当前原子任务，不能将测试轨迹的真值子任务当作端到端模型输出。
