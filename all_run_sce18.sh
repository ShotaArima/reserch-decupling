#!/bin/bash

# 出力ディレクトリの作成
OUTPUT_DIR="scenarios/scenario18_prophet_vs_sequential_vae/output"
mkdir -p "$OUTPUT_DIR"

# パラメータ定義
LOOKBACKS=(14 28)
PROPHET_MODELS=("p0_prophet" "p1_prophet_reg" "p2_prophet_segmented")
VAE_MODELS=("v1_seq_vae" "v2_seq_vae_transition")
ABLATIONS=("both" "common_only" "specific_only")
SEEDS=(42 52 62)

# --- 1. Prophet系モデルの実行 ---
for lb in "${LOOKBACKS[@]}"; do
  for model in "${PROPHET_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # ログファイル名の生成
      LOG_FILE="${OUTPUT_DIR}/log_lb${lb}_${model}_s${seed}.log"
      
      echo "Running: LB=$lb, Model=$model, Seed=$seed"
      
      uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py \
        --lookback "$lb" \
        --horizon 7 \
        --model "$model" \
        --seed "$seed" \
        2>&1 | tee "$LOG_FILE"
    done
  done
done

# --- 2. VAE系モデルの実行 (Ablationあり) ---
for lb in "${LOOKBACKS[@]}"; do
  for model in "${VAE_MODELS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        # ログファイル名の生成 (Ablationモードを含む)
        LOG_FILE="${OUTPUT_DIR}/log_lb${lb}_${model}_${ablation}_s${seed}.log"
        
        echo "Running: LB=$lb, Model=$model, Ablation=$ablation, Seed=$seed"
        
        uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py \
          --lookback "$lb" \
          --horizon 7 \
          --model "$model" \
          --ablation-mode "$ablation" \
          --seed "$seed" \
          2>&1 | tee "$LOG_FILE"
      done
    done
  done
done

# --- 3. V0 (Flatten VAE) の実行 (通常Ablationなしのため別枠) ---
for lb in "${LOOKBACKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/log_lb${lb}_v0_flatten_vae_s${seed}.log"
    
    uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py \
      --lookback "$lb" \
      --horizon 7 \
      --model v0_flatten_vae \
      --seed "$seed" \
      2>&1 | tee "$LOG_FILE"
  done
done