#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/ybpeng/miniconda3/envs/visrf-comm-cu130/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ybpeng/Data/ActualMulData/dataset_multimodal_data}"
GPU_ID="${GPU_ID:-0}"
RUN_ROOT="${RUN_ROOT:-experiments/latest_training_run}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"

ANCHOR_ROOT="${RUN_ROOT}/multiprocom_anchor"
FINAL_DIR="${RUN_ROOT}/multiprocom"
MODALITY_ROOT="${RUN_ROOT}/modality_baselines"
ABLATION_ROOT="${RUN_ROOT}/ablations"
GRU_AR_DIR="${ABLATION_ROOT}/wo_afsp_gru_ar"

run_fixed_epoch_method() {
  local method="$1"
  local output_root="$2"
  local output_dir="${output_root}/${method}"
  if [[ -s "${output_dir}/checkpoint_epoch140.pt" && -s "${output_dir}/results.json" ]]; then
    echo "[Skip] ${method}: completed output found at ${output_dir}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Incomplete output exists at ${output_dir}; remove it or choose a new RUN_ROOT." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    train_baseline_fixed_epoch_cross_scene.py \
    --method "${method}" \
    --root "${DATA_ROOT}" \
    --output-root "${output_root}" \
    --epochs 140 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda
}

train_multiprocom() {
  # Train the parallel multimodal anchor, then construct source-transition AFSP.
  run_fixed_epoch_method wo_afsp "${ANCHOR_ROOT}"
  if [[ -s "${FINAL_DIR}/selected_checkpoint.pt" && -s "${FINAL_DIR}/results.json" ]]; then
    echo "[Skip] MultiProCom AFSP: completed output found at ${FINAL_DIR}"
    return
  fi
  if [[ -e "${FINAL_DIR}" ]]; then
    echo "Incomplete output exists at ${FINAL_DIR}; remove it or choose a new RUN_ROOT." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    search_source_only_transition_afsp.py \
    --root "${DATA_ROOT}" \
    --checkpoint "${ANCHOR_ROOT}/wo_afsp/checkpoint_epoch140.pt" \
    --reference-results "${ANCHOR_ROOT}/wo_afsp/results.json" \
    --output-dir "${FINAL_DIR}" \
    --beta-override 0.4 \
    --beta-selected-on-target-validation \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda
}

train_modality_baselines() {
  run_fixed_epoch_method radar_only "${MODALITY_ROOT}"
  run_fixed_epoch_method vision_only "${MODALITY_ROOT}"
}

train_ablation_models() {
  # Remove RAMF or replace the proposed AFSP with the retained GRU-AR decoder.
  run_fixed_epoch_method wo_ramf "${ABLATION_ROOT}"
  if [[ -s "${GRU_AR_DIR}/best_checkpoint.pt" && -s "${GRU_AR_DIR}/results.json" ]]; then
    echo "[Skip] w/o proposed AFSP (GRU-AR): completed output found at ${GRU_AR_DIR}"
    return
  fi
  if [[ -e "${GRU_AR_DIR}" ]]; then
    echo "Incomplete output exists at ${GRU_AR_DIR}; remove it or choose a new RUN_ROOT." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    train_gru_ar_ablation_cross_scene.py \
    --root "${DATA_ROOT}" \
    --reference-summary "${ANCHOR_ROOT}/wo_afsp/summary.csv" \
    --output-dir "${GRU_AR_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda
}

echo "[1/3] Training MultiProCom"
train_multiprocom

echo "[2/3] Training modality baselines"
train_modality_baselines

echo "[3/3] Training ablation models"
train_ablation_models

echo "Final checkpoint: ${FINAL_DIR}/selected_checkpoint.pt"
echo "Modality baselines: ${MODALITY_ROOT}"
echo "Ablation models: ${ABLATION_ROOT}"
