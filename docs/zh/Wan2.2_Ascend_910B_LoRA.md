# Wan2.2 LoRA on Ascend 910B

本分支在 DiffSynth-Studio `4dbf980d4d0eb34eda136300dd0d72014cff8965`
上提供正确性优先的昇腾后端，首期覆盖：

- `Wan-AI/Wan2.2-TI2V-5B`
- `Wan-AI/Wan2.2-T2V-A14B` high-noise / low-noise
- `Wan-AI/Wan2.2-I2V-A14B` high-noise / low-noise

当前目标是 BF16 LoRA 前向、反向、保存和恢复。FP8、量化、CUDA Flash
Attention、SageAttention、xFormers 和图编译不属于首期范围。

## 1. 环境原则

优先使用已经包含配套 CANN、PyTorch 和 torch_npu 的昇腾官方容器。不要让
`pip install` 单独替换容器里的 `torch` 或 `torch_npu`；二者必须和 CANN、驱动成套。

进入容器后安装项目时，建议保留镜像内的 torch：

```bash
pip install -e . --no-deps
pip install torchvision transformers 'imageio[ffmpeg]' safetensors einops \
  modelscope ftfy pandas accelerate peft
```

如果使用项目的 `npu` extra，必须先确认 `pyproject.toml` 中的版本与目标镜像完全一致。

## 2. 运行环境门禁

```bash
export DIFFSYNTH_DEVICE=npu
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
python scripts/ascend_npu_smoke.py
```

探针必须通过 BF16 SDPA、BF16 Conv3D 和 FP32 AdamW 的前反向。失败时不要开始模型
训练，先检查驱动、CANN、torch_npu 版本矩阵和算子报错。

`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` 是精度基线。即使环境里意外装有 CUDA
Flash Attention，本分支也不会把 NPU tensor 传给 CUDA kernel。

## 3. 下载示例数据

```bash
modelscope download \
  --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include 'wanvideo/Wan2.2-TI2V-5B/*' \
  --include 'wanvideo/Wan2.2-T2V-A14B/*' \
  --include 'wanvideo/Wan2.2-I2V-A14B/*' \
  --local_dir ./data/diffsynth_example_dataset
```

## 4. 单卡最小训练

先分别执行 5B 的纯文本和图片条件分支：

```bash
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-t2v
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-i2v
```

再分别执行 A14B 的四个专家任务：

```bash
bash examples/wanvideo/ascend/train_lora_smoke.sh t2v-high
bash examples/wanvideo/ascend/train_lora_smoke.sh t2v-low
bash examples/wanvideo/ascend/train_lora_smoke.sh i2v-high
bash examples/wanvideo/ascend/train_lora_smoke.sh i2v-low
```

脚本默认使用 256×256、9 帧、rank 8、一个进程，只用于算子和训练闭环验证。可通过
环境变量修改：

```bash
HEIGHT=480 WIDTH=832 NUM_FRAMES=49 LORA_RANK=32 \
DATASET_REPEAT=100 NPROC_PER_NODE=8 \
bash examples/wanvideo/ascend/train_lora_smoke.sh t2v-high
```

不要从 8 卡生产配置开始排错。推荐顺序为：

1. 5B 单卡一个 step；
2. 5B 单卡 20～100 step 小数据过拟合；
3. T2V high、low 分别单卡一个 step；
4. I2V high、low 分别单卡一个 step；
5. DDP 多卡；
6. 显存不足后再引入 FSDP 或 CPU offload。

## 5. 精度对齐

CUDA 和 NPU 必须读取相同的预计算张量：text embedding、I2V image embedding /
image latent、target video latent、noise、timestep 和 LoRA 初始权重。

建议在模型关键位置注册 forward hook，并保存为字典：

```python
torch.save({
    "block_0": block_0_output.detach().cpu(),
    "block_20": block_20_output.detach().cpu(),
    "dit_output": dit_output.detach().cpu(),
    "loss": loss.detach().cpu(),
}, "alignment_cuda.pt")
```

在 NPU 保存同样的 key 后运行：

```bash
python scripts/compare_alignment_tensors.py \
  alignment_cuda.pt alignment_npu.pt \
  --min-cosine 0.995 \
  --max-relative-l2 0.02
```

首轮工程门槛：

| 项目 | 建议门槛 |
|---|---:|
| 单算子/中间层 cosine | ≥ 0.999 |
| BF16 DiT 最终输出 cosine | ≥ 0.995 |
| LoRA 梯度 cosine | ≥ 0.99 |
| 单步 loss 相对差 | ≤ 2% |
| 20-step 平均 loss 相对差 | ≤ 2% |

这些是 bring-up 门槛，不是最终质量标准。正确性稳定后可收紧到 DiT 输出
`≥0.998`、梯度 `≥0.995`、loss 相对差 `≤1%`。

### 双专家检查

T2V 和 I2V 必须独立检查 high/low：

- T2V boundary：`0.417`；
- I2V boundary：`0.358`；
- high 和 low 使用各自基础权重；
- 导出的两个 LoRA 不得互换；
- 边界测试先选区间内部 timestep，不要正好选切换点。

## 6. 已知边界

- 本地源码检查不能替代 910B 真机验证。
- PyTorch SDPA 是否覆盖目标 shape 取决于 torch_npu/CANN 版本。
- A14B 的激活显存可能要求多卡、FSDP、Context Parallel 或预缓存特征。
- CPU offload 已改为设备感知同步，但仍需要真机验证 pinned memory 与 NPU 异步拷贝。
- 原版 Unified Sequence Parallel 依赖 xFuser，不能假定 CUDA 实现可直接在 NPU 使用。
- 在单卡和 DDP 精度闭环完成前，不启用 NPU 融合 attention。

## 7. 真机验收记录

每次运行保存 upstream commit、CANN/驱动/torch/torch_npu 版本、910B 数量和显存、
模型与专家、分辨率和帧数、LoRA rank、逐 step loss、显存峰值、step time 以及
checkpoint 加载/保存结果。

第一次真机失败时，保留完整 traceback、`npu-smi info`、版本信息及所用命令，后续即可
针对具体算子继续收敛适配。
