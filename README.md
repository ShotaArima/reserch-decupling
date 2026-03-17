# reserch-decupling

FreshRetailNet-50K を使った Decoupling-inspired baseline 実装です。  
各シナリオをディレクトリで分離して、最小実験を独立実行できるようにしています。

## セットアップ（uv）

```bash
uv venv
source .venv/bin/activate
uv sync
```

## ディレクトリ構成

- `scenarios/scenario1_representation_probe`
  - global / local 表現分離の成立性を再構成で確認
- `scenarios/scenario2_raw_sales_forecast`
  - raw sales の 7 日先予測（最小）
- `scenarios/scenario3_latent_demand_recovery`
  - stockout mask を使った demand recovery（最小）
- `scenarios/scenario4_two_stage_pipeline`
  - recovery → forecasting の 2 段階
- `scenarios/scenario5_counterfactual_sanity`
  - local 固定 / global 交換（および逆）の反実仮想 sanity check
- `src/`
  - データロード、モデル、メトリクス共通部品

## 実行

```bash
uv run python scenarios/scenario1_representation_probe/run.py
uv run python scenarios/scenario2_raw_sales_forecast/run.py
uv run python scenarios/scenario3_latent_demand_recovery/run.py
uv run python scenarios/scenario4_two_stage_pipeline/run.py
uv run python scenarios/scenario5_counterfactual_sanity/run.py
```

> 注意: FreshRetailNet-50K の列名や split 名が将来変更された場合は、各スクリプト内の feature 定義を合わせて修正してください。
