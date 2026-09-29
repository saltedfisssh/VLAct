# VLAct RoboDojo smoke 验收记录

状态：通过。时间：2026-09-23 UTC。

- 仓库：`/kpfs_ssd/data/wzy/VLAct`，复用现有 `.venv`，未新建环境、未安装依赖。
- 模型：本地 VLAct_Qwen3_Pretrain 骨干 + 新初始化的 QwenOFT 14D 动作头；714 个骨干张量严格加载。
- 硬件：单张 A800 80GB；ZeRO-2、BF16、梯度检查点、优化器 CPU 卸载；45.087 亿参数设置为可训练。
- 数据：2 个任务，2 条训练轨迹（1107 帧），2 条验证轨迹（1204 帧）；训练从中随机采样。
- 实际执行：batch=1，梯度累积=2，2 次优化（4 个训练样本），另运行 1 个验证 batch。
- 两步训练 L1：0.698947、4.066269；验证 L1：3.549043，均为有限值。不同 batch 的两步 loss 不用于判断收敛。
- 训练峰值已分配显存：25.70 GiB；训练循环含验证与保存用时 162.3 秒，不含模型加载。
- 骨干抽样 65536 个元素中 13356 个发生变化；动作头抽样 65536 个元素均更新。
- VLAct 自身与 RoboDojo 预置 starVLA 模型代码均通过 `strict=True` 权重加载，并用真实三相机输入输出有限的 `[1,50,14]` 动作。
- 4 项数据契约测试通过，覆盖 q99 归一化、连续维度、轨迹/子任务边界、避免未来文本输入、训练/验证分离。
- 所有 shell 脚本通过 `bash -n`；Python 通过 `compileall`；8 卡命令完成 dry-run。

权重：`/kpfs_ssd/data/wzy/VLAct/results/RoboDojo/smoke_20260923_rgb_area/checkpoints/steps_2_pytorch_model.pt`

完整机器可读证据：[smoke_20260923_rgb_area/smoke_report.json](../../results/RoboDojo/smoke_20260923_rgb_area/smoke_report.json)。
训练日志：`results/RoboDojo/logs/smoke_20260923_rgb_area.log`。

全量数据准备也已运行：35×100=3500 条，按任务各 90/10 划分为 3150 条训练、350 条验证；1676319 / 183283 帧。源 LeRobot 数据不改写，统计只使用训练部分。

本次未执行 8 卡训练和仿真 rollout，不包含任务成功率；两步 smoke 权重只用于流程检查。默认动作基线不使用待验收的原子标签，子任务条件模式已实现但未进行正式标注训练。

8 卡启动命令与路径覆盖、恢复训练、子任务条件模式见 [README.md](README.md)。
