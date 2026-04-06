# reserch-decupling

FreshRetailNet-50K を使った Decoupling-inspired baseline 実装です。  
各シナリオをディレクトリで分離して、最小実験を独立実行できるようにしています。

## セットアップ（uv）

```bash
uv venv
source .venv/bin/activate
uv sync
```

## まず実行前に確認すること

いきなりシナリオスクリプトを実行しても動く設計ですが、初回は以下の順序を推奨します。

1. **依存関係が入っているか確認**

```bash
uv run python -c "import torch, datasets, pandas, numpy; print('ok')"
```

2. **ネットワーク経由で Hugging Face にアクセスできるか確認**

```bash
 uv run python -c 'from datasets import load_dataset; dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K"); print(dataset)' 
```

3. 問題なければシナリオを実行

> `load_dataset(...)` は初回実行時にデータを自動ダウンロードしてキャッシュします。  
> そのため、**別途の手動セットアップは不要**ですが、ネットワーク制限がある環境では失敗する可能性があります。

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
- `scenarios/scenario6_local_global_ablation`
  - local only / global only / both の予測アブレーション
- `scenarios/scenario7_counterfactual_sanity_denormalized`
  - counterfactual を denormalize 後に再評価して継続可否を判断
- `src/`
  - データロード、モデル、メトリクス共通部品

## 実行

```bash
uv run python scenarios/scenario1_representation_probe/run.py
uv run python scenarios/scenario2_raw_sales_forecast/run.py
uv run python scenarios/scenario3_latent_demand_recovery/run.py
uv run python scenarios/scenario4_two_stage_pipeline/run.py
uv run python scenarios/scenario5_counterfactual_sanity/run.py
uv run python scenarios/scenario6_local_global_ablation/run.py
uv run python scenarios/scenario7_counterfactual_sanity_denormalized/run.py
```

> 注意: FreshRetailNet-50K の列名や split 名が将来変更された場合は、各スクリプト内の feature 定義を合わせて修正してください。
