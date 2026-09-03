# Wan2.2 LoRA 昇腾 910B 操作说明书

本说明书覆盖从镜像构建、容器启动、模型与数据下载，到 Wan2.2 LoRA 冒烟训练和结果验收的完整流程。适用训练目标如下：

| 模型 | 任务 | 训练单元 |
| --- | --- | --- |
| Wan2.2-TI2V-5B | T2V | 1 个 LoRA |
| Wan2.2-TI2V-5B | I2V | 1 个 LoRA |
| Wan2.2-T2V-A14B | T2V | high-noise、low-noise 各 1 个 LoRA |
| Wan2.2-I2V-A14B | I2V | high-noise、low-noise 各 1 个 LoRA |

建议先完成 TI2V-5B T2V 冒烟训练，再依次执行 TI2V-5B I2V、T2V-A14B 和 I2V-A14B。

## 1. 目录约定

宿主机工作根目录：

```bash
export WORK_ROOT=/path/to/workspace
export REPO_ROOT=${WORK_ROOT}/DiffSynth-Studio
export MODEL_ROOT=${WORK_ROOT}/models
export OUTPUT_ROOT=${WORK_ROOT}/outputs
export LOG_ROOT=${WORK_ROOT}/logs

mkdir -p "${MODEL_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}" "${REPO_ROOT}/data"
```

本文后续命令均使用以上目录。

## 2. 进入源码目录

```bash
cd "${REPO_ROOT}"
test -f UPSTREAM_COMMIT
```

源码下载与补丁安装按照补丁包根目录的 `README.md` 完成。

## 3. 构建配套镜像

镜像基于以下 Quay 基础镜像：

```text
quay.io/ascend/mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11
```

仓库中的 `docker/Dockerfile.ascend` 固定了以下框架组合：

```text
torch       2.7.1
torch-npu   2.7.1.post8
torchvision 0.22.1
torchaudio  2.7.1
```

它同时安装训练依赖、音视频系统库、DiffSynth-Studio-Ascend，并在构建阶段执行版本检查和 Wan RoPE 回归测试。

构建镜像：

```bash
cd "${REPO_ROOT}"
docker build --pull \
  -f docker/Dockerfile.ascend \
  -t diffsynth-wan22-ascend:torch2.7.1-cann9.1 .
```

构建成功的最后阶段会显示：

```text
DiffSynth Ascend framework versions: OK
Ran 4 tests
OK
```

## 4. 启动训练容器

```bash
docker run --rm -it \
  --name diffsynth-wan22 \
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
  -v "${MODEL_ROOT}:/workspace/models" \
  -v "${OUTPUT_ROOT}:/workspace/outputs" \
  -v "${LOG_ROOT}:/workspace/logs" \
  -v "${REPO_ROOT}/data:/workspace/DiffSynth-Studio-Ascend/data" \
  diffsynth-wan22-ascend:torch2.7.1-cann9.1
```

进入容器后的代码目录为：

```text
/workspace/DiffSynth-Studio-Ascend
```

## 5. 验证 NPU 运行环境

在容器内执行：

```bash
cd /workspace/DiffSynth-Studio-Ascend
npu-smi info
python scripts/ascend_npu_smoke.py
python tests/test_ascend_wan_rope.py
```

验收结果：

```text
PASS: BF16 SDPA
PASS: BF16 Conv3D
PASS: FP32 AdamW forward/backward
Ran 4 tests
OK
```

## 6. 下载模型、Tokenizer 和示例数据

在容器内设置统一模型目录：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models
export DATA_BASE_PATH=/workspace/DiffSynth-Studio-Ascend/data/diffsynth_example_dataset
```

下载 TI2V-5B 及其数据：

```bash
bash scripts/download_wan22_lora_assets.sh ti2v-5b
```

下载 T2V-A14B 及其数据：

```bash
bash scripts/download_wan22_lora_assets.sh t2v-a14b
```

下载 I2V-A14B 及其数据：

```bash
bash scripts/download_wan22_lora_assets.sh i2v-a14b
```

每条命令完成后均输出：

```text
Wan2.2 <profile> assets: OK
```

下载脚本会完整保存对应模型仓库，并准备：

- DiT 权重；
- T5 权重；
- VAE 权重；
- UMT5 Tokenizer；
- 对应任务的示例数据和 `metadata.csv`。

## 7. 设置训练环境

在容器内执行：

```bash
cd /workspace/DiffSynth-Studio-Ascend

export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models
export DIFFSYNTH_SKIP_DOWNLOAD=True
export DIFFSYNTH_DEVICE=npu
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=1800
```

训练使用 BF16 模型计算和 FP32 AdamW 优化器状态。模型、数据和输出路径在所有训练命令中保持一致。

## 8. TI2V-5B T2V 冒烟训练

```bash
NPROC_PER_NODE=1 \
HEIGHT=256 \
WIDTH=256 \
NUM_FRAMES=9 \
LORA_RANK=8 \
DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-t2v \
  2>&1 | tee /workspace/logs/ti2v-5b-t2v.log
```

验收训练输出：

```bash
python scripts/verify_lora_run.py \
  /workspace/outputs/ti2v-5b-t2v
```

通过时输出：

```text
LoRA checkpoints: <数量>
Loss records: <数量>
Loss range: <最小值> .. <最大值>
LoRA run verification: PASS
```

## 9. TI2V-5B I2V 冒烟训练

```bash
NPROC_PER_NODE=1 \
HEIGHT=256 \
WIDTH=256 \
NUM_FRAMES=9 \
LORA_RANK=8 \
DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-i2v \
  2>&1 | tee /workspace/logs/ti2v-5b-i2v.log

python scripts/verify_lora_run.py \
  /workspace/outputs/ti2v-5b-i2v
```

## 10. T2V-A14B 冒烟训练

Wan2.2 T2V-A14B 由 high-noise 和 low-noise 两个 DiT 组成，因此分别训练两个 LoRA。

```bash
NPROC_PER_NODE=1 HEIGHT=256 WIDTH=256 NUM_FRAMES=9 LORA_RANK=8 DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh t2v-high \
  2>&1 | tee /workspace/logs/t2v-high.log

python scripts/verify_lora_run.py \
  /workspace/outputs/t2v-high
```

```bash
NPROC_PER_NODE=1 HEIGHT=256 WIDTH=256 NUM_FRAMES=9 LORA_RANK=8 DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh t2v-low \
  2>&1 | tee /workspace/logs/t2v-low.log

python scripts/verify_lora_run.py \
  /workspace/outputs/t2v-low
```

## 11. I2V-A14B 冒烟训练

Wan2.2 I2V-A14B 同样分别训练 high-noise 和 low-noise 两个 LoRA。

```bash
NPROC_PER_NODE=1 HEIGHT=256 WIDTH=256 NUM_FRAMES=9 LORA_RANK=8 DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh i2v-high \
  2>&1 | tee /workspace/logs/i2v-high.log

python scripts/verify_lora_run.py \
  /workspace/outputs/i2v-high
```

```bash
NPROC_PER_NODE=1 HEIGHT=256 WIDTH=256 NUM_FRAMES=9 LORA_RANK=8 DATASET_REPEAT=1 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh i2v-low \
  2>&1 | tee /workspace/logs/i2v-low.log

python scripts/verify_lora_run.py \
  /workspace/outputs/i2v-low
```

## 12. 正式训练参数

冒烟训练全部通过后，沿用相同脚本并调整训练规模：

```bash
NPROC_PER_NODE=1 \
HEIGHT=480 \
WIDTH=832 \
NUM_FRAMES=49 \
LORA_RANK=32 \
DATASET_REPEAT=100 \
NUM_EPOCHS=5 \
SAVE_STEPS=100 \
LORA_TARGET_MODULES=q,k,v,o,ffn.0,ffn.2 \
OUTPUT_ROOT=/workspace/outputs \
bash examples/wanvideo/ascend/train_lora_smoke.sh ti2v-5b-t2v \
  2>&1 | tee /workspace/logs/ti2v-5b-t2v-train.log
```

将最后一个参数替换为相应训练单元即可执行其余任务：

```text
ti2v-5b-i2v
t2v-high
t2v-low
i2v-high
i2v-low
```

每个正式训练任务结束后执行 `scripts/verify_lora_run.py`，其 `PASS` 结果作为本次训练产物验收记录。

## 13. 完整验收标准

每个训练单元必须同时满足：

- 训练进程退出码为 0；
- 输出目录存在非空 LoRA `safetensors`；
- LoRA 文件包含张量，且全部张量为有限数值；
- `loss.csv` 存在有效 loss 记录；
- 所有 loss 均为有限数值；
- `scripts/verify_lora_run.py` 输出 `LoRA run verification: PASS`。
