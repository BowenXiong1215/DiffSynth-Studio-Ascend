#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-ti2v-5b}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(pwd)/models}"
DATA_BASE_PATH="${DATA_BASE_PATH:-data/diffsynth_example_dataset}"

case "${PROFILE}" in
  ti2v-5b)
    MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
    DATASET_NAME="Wan2.2-TI2V-5B"
    VAE_NAME="Wan2.2_VAE.pth"
    ;;
  t2v-a14b)
    MODEL_ID="Wan-AI/Wan2.2-T2V-A14B"
    DATASET_NAME="Wan2.2-T2V-A14B"
    VAE_NAME="Wan2.1_VAE.pth"
    ;;
  i2v-a14b)
    MODEL_ID="Wan-AI/Wan2.2-I2V-A14B"
    DATASET_NAME="Wan2.2-I2V-A14B"
    VAE_NAME="Wan2.1_VAE.pth"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    echo "Expected: ti2v-5b, t2v-a14b or i2v-a14b" >&2
    exit 2
    ;;
esac

MODEL_DIR="${MODEL_BASE_PATH}/${MODEL_ID}"
TOKENIZER_DIR="${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B"

mkdir -p "${MODEL_DIR}" "${TOKENIZER_DIR}" "${DATA_BASE_PATH}"

modelscope download --model "${MODEL_ID}" --local_dir "${MODEL_DIR}"
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local_dir "${TOKENIZER_DIR}"
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "wanvideo/${DATASET_NAME}/*" \
  --local_dir "${DATA_BASE_PATH}"

test -f "${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
test -f "${MODEL_DIR}/${VAE_NAME}"
test -f "${TOKENIZER_DIR}/google/umt5-xxl/tokenizer_config.json"
test -f "${DATA_BASE_PATH}/wanvideo/${DATASET_NAME}/metadata.csv"

case "${PROFILE}" in
  ti2v-5b)
    compgen -G "${MODEL_DIR}/diffusion_pytorch_model*.safetensors" >/dev/null
    ;;
  *)
    compgen -G "${MODEL_DIR}/high_noise_model/diffusion_pytorch_model*.safetensors" >/dev/null
    compgen -G "${MODEL_DIR}/low_noise_model/diffusion_pytorch_model*.safetensors" >/dev/null
    ;;
esac

echo "Wan2.2 ${PROFILE} assets: OK"
