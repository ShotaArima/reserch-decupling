#!/bin/bash

# 設定
OUT="commands_with_logs.txt"
LOG_DIR="scenarios/scenario18_prophet_vs_sequential_vae/output"

# パラメータ
LOOKBACKS=(14 28)
PROPHET_MODELS=("p0_prophet" "p1_prophet_reg" "p2_prophet_segmented")
VAE_MODELS=("v1_seq_vae" "v2_seq_vae_transition")
ABLATIONS=("both" "common_only" "specific_only")
SEEDS=(42 52 62)

echo "# === Commands with Log Redirection (tee) ===" > "$OUT"

# 1. Prophet系
echo -e "\n# --- 1. Prophet Models ---" >> "$OUT"
for lb in "${LOOKBACKS[@]}"; do
  for model in "${PROPHET_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      LOG_FILE="${LOG_DIR}/log_lb${lb}_${model}_s${seed}.log"
      CMD="uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback $lb --horizon 7 --model $model --seed $seed"
      echo "$CMD 2>&1 | tee $LOG_FILE" >> "$OUT"
    done
  done
done

# 2. VAE系 (Ablationあり)
echo -e "\n# --- 2. Sequential VAE Models ---" >> "$OUT"
for lb in "${LOOKBACKS[@]}"; do
  for model in "${VAE_MODELS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        LOG_FILE="${LOG_DIR}/log_lb${lb}_${model}_${ablation}_s${seed}.log"
        CMD="uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback $lb --horizon 7 --model $model --ablation-mode $ablation --seed $seed"
        echo "$CMD 2>&1 | tee $LOG_FILE" >> "$OUT"
      done
    done
  done
done

# 3. V0 (Flatten VAE)
echo -e "\n# --- 3. V0 Flatten VAE ---" >> "$OUT"
for lb in "${LOOKBACKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    LOG_FILE="${LOG_DIR}/log_lb${lb}_v0_flatten_vae_s${seed}.log"
    CMD="uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback $lb --horizon 7 --model v0_flatten_vae --seed $seed"
    echo "$CMD 2>&1 | tee $LOG_FILE" >> "$OUT"
  done
done

echo "Done. Created $OUT"
