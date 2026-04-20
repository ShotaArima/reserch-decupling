# 2の残り
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v2_seq_vae_transition --ablation-mode common_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v2_seq_vae_transition_common_only_s62.log
echo "[end] : log_lb14_v2_seq_vae_transition_common_only_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v2_seq_vae_transition_specific_only_s42.log
echo "[end] : log_lb14_v2_seq_vae_transition_specific_only_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v2_seq_vae_transition_specific_only_s52.log
echo "[end] : log_lb14_v2_seq_vae_transition_specific_only_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v2_seq_vae_transition_specific_only_s62.log
echo "[end] : log_lb14_v2_seq_vae_transition_specific_only_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode both --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_both_s42.log
echo "[end] : log_lb28_v1_seq_vae_both_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode both --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_both_s52.log
echo "[end] : log_lb28_v1_seq_vae_both_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode both --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_both_s62.log
echo "[end] : log_lb28_v1_seq_vae_both_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode common_only --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_common_only_s42.log
echo "[end] : log_lb28_v1_seq_vae_common_only_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode common_only --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_common_only_s52.log
echo "[end] : log_lb28_v1_seq_vae_common_only_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode common_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_common_only_s62.log
echo "[end] : log_lb28_v1_seq_vae_common_only_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode specific_only --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_specific_only_s42.log
echo "[end] : log_lb28_v1_seq_vae_specific_only_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode specific_only --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_specific_only_s52.log
echo "[end] : log_lb28_v1_seq_vae_specific_only_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v1_seq_vae --ablation-mode specific_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v1_seq_vae_specific_only_s62.log
echo "[end] : log_lb28_v1_seq_vae_specific_only_s62"　
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode both --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_both_s42.log
echo "[end] : log_lb28_v2_seq_vae_transition_both_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode both --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_both_s52.log
echo "[end] : log_lb28_v2_seq_vae_transition_both_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode both --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_both_s62.log
echo "[end] : log_lb28_v2_seq_vae_transition_both_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode common_only --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_common_only_s42.log
echo "[end] : log_lb28_v2_seq_vae_transition_common_only_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode common_only --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_common_only_s52.log
echo "[end] : log_lb28_v2_seq_vae_transition_common_only_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode common_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_common_only_s62.log
echo "[end] : log_lb28_v2_seq_vae_transition_common_only_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_specific_only_s42.log
echo "[end] : log_lb28_v2_seq_vae_transition_specific_only_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_specific_only_s52.log
echo "[end] : log_lb28_v2_seq_vae_transition_specific_only_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v2_seq_vae_transition --ablation-mode specific_only --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v2_seq_vae_transition_specific_only_s62.log
echo "[end] : log_lb28_v2_seq_vae_transition_specific_only_s62"

# 3
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s42.log
echo "[end] : log_lb14_v0_flatten_vae_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s52.log
echo "[end] : log_lb14_v0_flatten_vae_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s62.log
echo "[end] : log_lb14_v0_flatten_vae_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s42.log
echo "[end] : log_lb28_v0_flatten_vae_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s52.log
echo "[end] : log_lb28_v0_flatten_vae_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s62.log
echo "[end] : log_lb28_v0_flatten_vae_s62"