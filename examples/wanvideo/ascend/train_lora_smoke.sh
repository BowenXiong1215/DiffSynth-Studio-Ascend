#!/usr/bin/env bash
set -euo pipefail

# Correctness-first Wan2.2 LoRA smoke tests for Ascend 910B.
# Usage: bash examples/wanvideo/ascend/train_lora_smoke.sh MODEL
# MODEL: ti2v-5b-t2v | ti2v-5b-i2v | t2v-high | t2v-low | i2v-high | i2v-low

MODEL="${1:-ti2v-5b-t2v}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_FRAMES="${NUM_FRAMES:-9}"
LORA_RANK="${LORA_RANK:-8}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATA_ROOT="${DATA_ROOT:-data/diffsynth_example_dataset/wanvideo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-models/train/ascend_smoke}"

export DIFFSYNTH_DEVICE="${DIFFSYNTH_DEVICE:-npu}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"

case "${MODEL}" in
  ti2v-5b-t2v)
    DATASET_NAME="Wan2.2-TI2V-5B"
    MODEL_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
    EXTRA_ARGS=(--max_timestep_boundary 1 --min_timestep_boundary 0)
    ;;
  ti2v-5b-i2v)
    DATASET_NAME="Wan2.2-TI2V-5B"
    MODEL_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
    EXTRA_ARGS=(--extra_inputs input_image --max_timestep_boundary 1 --min_timestep_boundary 0)
    ;;
  t2v-high)
    DATASET_NAME="Wan2.2-T2V-A14B"
    MODEL_PATHS="Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth"
    EXTRA_ARGS=(--max_timestep_boundary 0.417 --min_timestep_boundary 0)
    ;;
  t2v-low)
    DATASET_NAME="Wan2.2-T2V-A14B"
    MODEL_PATHS="Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth"
    EXTRA_ARGS=(--max_timestep_boundary 1 --min_timestep_boundary 0.417)
    ;;
  i2v-high)
    DATASET_NAME="Wan2.2-I2V-A14B"
    MODEL_PATHS="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
    EXTRA_ARGS=(--extra_inputs input_image --max_timestep_boundary 0.358 --min_timestep_boundary 0)
    ;;
  i2v-low)
    DATASET_NAME="Wan2.2-I2V-A14B"
    MODEL_PATHS="Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
    EXTRA_ARGS=(--extra_inputs input_image --max_timestep_boundary 1 --min_timestep_boundary 0.358)
    ;;
  *)
    echo "Unknown MODEL '${MODEL}'. Expected ti2v-5b-t2v, ti2v-5b-i2v, t2v-high, t2v-low, i2v-high or i2v-low." >&2
    exit 2
    ;;
esac

DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}"
METADATA_PATH="${DATASET_PATH}/metadata.csv"
if [[ ! -f "${METADATA_PATH}" ]]; then
  echo "Dataset metadata not found: ${METADATA_PATH}" >&2
  echo "Download it with:" >&2
  echo "modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include 'wanvideo/${DATASET_NAME}/*' --local_dir ./data/diffsynth_example_dataset" >&2
  exit 1
fi

accelerate launch --num_processes "${NPROC_PER_NODE}" \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_PATH}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers 0 \
  --model_id_with_origin_paths "${MODEL_PATHS}" \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --num_epochs 1 \
  --save_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_ROOT}/${MODEL}" \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o" \
  --lora_rank "${LORA_RANK}" \
  "${EXTRA_ARGS[@]}"
