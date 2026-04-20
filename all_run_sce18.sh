echo "0. [start]"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s42.log
echo "1. [end] log_lb14_v0_flatten_vae_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s52.log
echo "2. [end] log_lb14_v0_flatten_vae_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 14 --horizon 7 --model v0_flatten_vae --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb14_v0_flatten_vae_s62.log
echo "3. [end] log_lb14_v0_flatten_vae_s62"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 42 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s42.log
echo "4. [end] log_lb28_v0_flatten_vae_s42"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 52 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s52.log
echo "5. [end] log_lb28_v0_flatten_vae_s52"
uv run python -u scenarios/scenario18_prophet_vs_sequential_vae/run.py --lookback 28 --horizon 7 --model v0_flatten_vae --seed 62 2>&1 | tee scenarios/scenario18_prophet_vs_sequential_vae/output/log_lb28_v0_flatten_vae_s62.log
echo "6. [end] log_lb28_v0_flatten_vae_s62"