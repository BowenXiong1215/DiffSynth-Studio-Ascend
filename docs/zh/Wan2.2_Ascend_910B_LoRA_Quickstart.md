# Wan2.2 LoRA 昇腾 910B 部署与启动手册

本文是一份可以直接照着执行的操作手册，覆盖从基础镜像、容器、Python 环境、
DiffSynth-Studio 代码、ModelScope 权重与数据，到第一次 Wan2.2 TI2V-5B LoRA
训练及结果验收的完整流程。

首轮目标是正确性闭环，不追求速度和生成质量。建议严格按以下顺序操作：

1. 单卡、BF16、PyTorch SDPA；
2. 无权重 NPU smoke；
3. TI2V-5B 文生视频 LoRA 一个 step；
4. TI2V-5B 图生视频 LoRA 一个 step；
5. 小数据过拟合；
6. 最后再扩展到 A14B、DDP 和性能优化。

## 1. 目录约定

本文统一使用以下目录：

```text
/hpc-to-ds-0115/x00876811/
├── DiffSynth-Studio-Ascend/    # 代码
├── models/                     # 模型权重
├── outputs/                    # LoRA checkpoint
└── logs/                       # 日志和环境记录
```

创建目录：

```bash
mkdir -p /hpc-to-ds-0115/x00876811/{models,outputs,logs}
```

建议至少准备 100 GB 空间用于 TI2V-5B；如果后续还要下载 T2V-A14B 和
I2V-A14B，建议预留 200～300 GB：

```bash
df -h /hpc-to-ds-0115/x00876811
```

## 2. 检查宿主机

在宿主机执行：

```bash
uname -m
npu-smi info
docker version
cat /usr/local/Ascend/driver/version.info
```

`npu-smi info` 必须能看到 910B 设备。本文使用的容器是 CANN 9.1.0，宿主机
驱动必须与它兼容。

## 3. 拉取基础镜像

使用已经过 Quay Registry manifest 验证的多架构镜像：

```bash
docker pull \
  quay.io/ascend/mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11
```

该镜像包含 CANN 9.1.0、PyTorch 2.7.1、TorchNPU 2.7.1.post8，并同时提供
`linux/amd64` 和 `linux/arm64` manifest。

## 4. 启动单卡开发容器

以下示例只透传 0 号 NPU。多卡验证应在单卡闭环通过后进行。

```bash
docker run --rm -it \
  --name diffsynth-wan22-dev \
  --network=host \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /hpc-to-ds-0115/x00876811:/hpc-to-ds-0115/x00876811 \
  quay.io/ascend/mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11 \
  bash
```

如果运行平台已经自动透传 NPU 和挂载数据目录，可以跳过相应参数，但容器内必须
能访问 `/dev/davinci0` 和上述工作目录。

## 5. 验证基础镜像

进入容器后执行：

```bash
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY
```

期望看到 PyTorch 2.7.1、TorchNPU 2.7.1.post8，并且 `NPU available` 为
`True`。

## 6. 准备代码

推荐直接克隆最新仓库：

```bash
cd /hpc-to-ds-0115/x00876811
git clone https://github.com/BowenXiong1215/DiffSynth-Studio-Ascend.git
cd DiffSynth-Studio-Ascend
```

如果使用压缩包，解压后应确保只有一层代码目录：

```text
/hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend/pyproject.toml
```

而不是：

```text
/hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend/DiffSynth-Studio-Ascend/pyproject.toml
```

如果出现双层目录，在外层目录执行：

```bash
cd /hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend
shopt -s dotglob nullglob
mv DiffSynth-Studio-Ascend/* .
shopt -u dotglob nullglob
rmdir DiffSynth-Studio-Ascend
```

## 7. 安装 Python 依赖

### 7.1 安装 DiffSynth 本身

使用 editable、无依赖、无隔离方式安装，避免 pip 覆盖镜像内配套的 PyTorch 和
TorchNPU：

```bash
cd /hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend
python -m pip install -vvv --no-build-isolation --no-deps -e .
```

### 7.2 锁住 NPU 框架版本

创建约束文件：

```bash
cat >/tmp/diffsynth-npu-constraints.txt <<'EOF'
torch==2.7.1
torch-npu==2.7.1.post8
torchvision==0.22.1
torchaudio==2.7.1
EOF
```

安装运行依赖。约束文件的作用是：如果某个包要求升级 PyTorch，pip 应直接报冲突，
而不是静默换成 CUDA 版本。

```bash
python -m pip install -v --progress-bar on \
  -c /tmp/diffsynth-npu-constraints.txt \
  transformers \
  "imageio[ffmpeg]" \
  safetensors \
  einops \
  modelscope \
  ftfy \
  pandas \
  accelerate \
  peft \
  sentencepiece \
  librosa
```

如果需要安装 `torchvision` 或 `torchaudio`，必须与 PyTorch 2.7.1 配套。x86_64
使用 CPU wheel：

```bash
if [ "$(uname -m)" = "x86_64" ]; then
  python -m pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    "torchvision==0.22.1+cpu" \
    "torchaudio==2.7.1+cpu"
else
  python -m pip install --force-reinstall --no-deps \
    "torchvision==0.22.1" \
    "torchaudio==2.7.1"
fi
```

不要为了解决 `libcudart.so` 缺失而安装 CUDA。出现 `libcudart.so.13` 通常意味着
误装了 CUDA 版 PyTorch、torchvision 或 torchaudio，应恢复为上述 CPU/NPU 配套
版本。

安装后检查：

```bash
python -m pip check

python - <<'PY'
import torch
import torch_npu
import torchvision
import torchaudio
import transformers
import librosa

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("librosa:", librosa.__version__)
print("NPU available:", torch.npu.is_available())
PY
```

## 8. 运行无权重 NPU smoke

```bash
cd /hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend

DIFFSYNTH_DEVICE=npu \
DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch \
python scripts/ascend_npu_smoke.py
```

必须看到类似输出：

```text
PASS: BF16 SDPA, BF16 Conv3D and FP32 AdamW forward/backward
```

该步骤不需要模型权重。如果失败，不要继续模型训练。

## 9. 下载 TI2V-5B 权重

统一设置模型根目录：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/hpc-to-ds-0115/x00876811/models
mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH"
```

下载 TI2V-5B DiT：

```bash
modelscope download \
  --model Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local_dir "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"
```

下载公共 T5 和 Wan2.2 VAE：

```bash
modelscope download \
  --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --include 'models_t5_umt5-xxl-enc-bf16.safetensors' \
  --include 'Wan2.2_VAE.safetensors' \
  --local_dir "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
```

下载 tokenizer：

```bash
modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --include 'google/umt5-xxl/*' \
  --local_dir "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B"
```

## 10. 下载 TI2V-5B 示例数据

在仓库根目录执行：

```bash
cd /hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend

modelscope download \
  --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include 'wanvideo/Wan2.2-TI2V-5B/*' \
  --local_dir ./data/diffsynth_example_dataset
```

确认数据：

```bash
ls -lh data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv
```

## 11. 检查本地文件布局

启动训练前应至少具备：

```text
/hpc-to-ds-0115/x00876811/models/
├── Wan-AI/
│   ├── Wan2.2-TI2V-5B/
│   │   └── diffusion_pytorch_model*.safetensors
│   └── Wan2.1-T2V-1.3B/
│       └── google/umt5-xxl/
└── DiffSynth-Studio/
    └── Wan-Series-Converted-Safetensors/
        ├── models_t5_umt5-xxl-enc-bf16.safetensors
        └── Wan2.2_VAE.safetensors
```

可以用以下命令快速检查：

```bash
find /hpc-to-ds-0115/x00876811/models \
  -type f \
  \( -name '*.safetensors' -o -name 'tokenizer_config.json' -o -name 'spiece.model' \) \
  | sort
```

## 12. 启动 TI2V-5B 文生视频 LoRA

设置运行环境：

```bash
cd /hpc-to-ds-0115/x00876811/DiffSynth-Studio-Ascend

export DIFFSYNTH_MODEL_BASE_PATH=/hpc-to-ds-0115/x00876811/models
export DIFFSYNTH_SKIP_DOWNLOAD=True
export DIFFSYNTH_DEVICE=npu
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=1800

mkdir -p /hpc-to-ds-0115/x00876811/{outputs/ascend_smoke,logs}
```

启动单卡、256×256、9 帧、rank 8 的最小训练：

```bash
set -o pipefail

NPROC_PER_NODE=1 \
HEIGHT=256 \
WIDTH=256 \
NUM_FRAMES=9 \
LORA_RANK=8 \
DATASET_REPEAT=1 \
OUTPUT_ROOT=/hpc-to-ds-0115/x00876811/outputs/ascend_smoke \
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-t2v \
2>&1 | tee /hpc-to-ds-0115/x00876811/logs/ti2v-5b-t2v.log
```

训练过程中可以在另一个终端观察：

```bash
watch -n 2 npu-smi info
```

## 13. 验收 LoRA 结果

查看 checkpoint：

```bash
find /hpc-to-ds-0115/x00876811/outputs/ascend_smoke/ti2v-5b-t2v \
  -type f -ls
```

启动脚本默认启用 CSV loss 日志。查看 loss：

```bash
cat /hpc-to-ds-0115/x00876811/outputs/ascend_smoke/ti2v-5b-t2v/loss.csv
```

检查 checkpoint 中所有 LoRA 张量是否有限：

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import torch

root = Path("/hpc-to-ds-0115/x00876811/outputs/ascend_smoke/ti2v-5b-t2v")
files = sorted(root.rglob("*.safetensors"))
assert files, "没有找到 LoRA checkpoint"

for path in files:
    with safe_open(path, framework="pt", device="cpu") as f:
        bad = [key for key in f.keys() if not torch.isfinite(f.get_tensor(key)).all()]
    assert not bad, f"{path} 包含 nan/inf: {bad}"
    print("PASS:", path)
PY
```

本阶段的通过标准：

- 训练命令正常退出；
- 至少生成一个非空 `.safetensors` checkpoint；
- `loss.csv` 中存在有限 loss；
- checkpoint 中没有 `nan` 或 `inf`。

## 14. 下一步

TI2V-5B 文生视频通过后，使用同一套模型和数据验证图生视频分支：

```bash
NPROC_PER_NODE=1 \
HEIGHT=256 \
WIDTH=256 \
NUM_FRAMES=9 \
LORA_RANK=8 \
DATASET_REPEAT=1 \
OUTPUT_ROOT=/hpc-to-ds-0115/x00876811/outputs/ascend_smoke \
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-i2v \
2>&1 | tee /hpc-to-ds-0115/x00876811/logs/ti2v-5b-i2v.log
```

两个分支均通过后，将 `DATASET_REPEAT` 提高到 20～100 做小数据过拟合，确认
loss 总体下降。之后再依次推进 T2V-A14B high/low、I2V-A14B high/low、多卡
DDP 和 CUDA/NPU 精度对齐。

## 15. 保存成功环境

第一次成功后立即保存环境，以便重建镜像：

```bash
python -m pip freeze \
  > /hpc-to-ds-0115/x00876811/logs/success-requirements.txt

npu-smi info \
  > /hpc-to-ds-0115/x00876811/logs/success-npu-info.txt
```

后续生产镜像应以这份成功环境为基础锁定依赖，不再使用无版本限制的
`pip install -U torch torchvision torchaudio`。

## 16. 常见问题

### `Installing build dependencies` 长时间无进度

```bash
python -m pip install -vvv --no-index --no-build-isolation --no-deps -e .
```

### `libcudart.so.13` 不存在

说明误装了 CUDA 版 PyTorch、torchvision 或 torchaudio。不要安装 CUDA runtime，
应恢复 PyTorch 2.7.1 对应的 CPU wheel，并保留 TorchNPU 2.7.1.post8。

### 找不到公共 T5 或 VAE

检查：

```bash
find /hpc-to-ds-0115/x00876811/models \
  -type f \
  -name 'models_t5_umt5-xxl-enc-bf16*' \
  -o -name 'Wan2.2_VAE*'
```

公共文件必须位于：

```text
/hpc-to-ds-0115/x00876811/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/
```

### 最后只看到 `CalledProcessError`

`CalledProcessError` 只是 Accelerate 对子进程失败的汇总。应从日志中寻找最早出现的
`Traceback` 和第一条实际异常。

