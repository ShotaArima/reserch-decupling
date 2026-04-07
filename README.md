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
- `scenarios/scenario8_recovery_subset_diagnosis`
  - A3: raw/Scenario2/Scenario4 を all・stockout・non-stockout で subset 評価
- `scenarios/scenario9_common_specific_feature_assignment`
  - common/specific 特徴量一次割り当て（Exp-0/1/2 + ablation）
- `scenarios/scenario10_stock_extension_local_branch`
  - stock 系特徴を specific 側へ追加する拡張検証（Exp-3 系）
- `scenarios/scenario17_horizon_role_gap`
  - 予測 horizon（1/3/7-step）で common/specific の寄与差を比較
- `doc/08-scenario8_recovery_subset_diagnosis.md`
  - Scenario8 の実験設計ドキュメント
- `doc/09-scenario9_common_specific_feature_assignment.md`
  - Scenario9 の共通/固有一次割り当て実験計画
- `doc/10-scenario10_stock_extension_for_local_branch.md`
  - Scenario10 の stock 状態変数 local 拡張実験計画
- `doc/17-scenario17_horizon_role_gap.md`
  - Scenario17 の horizon 依存役割差実験計画
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
uv run python scenarios/scenario8_recovery_subset_diagnosis/run.py
uv run python scenarios/scenario9_common_specific_feature_assignment/run.py
uv run python scenarios/scenario10_stock_extension_local_branch/run.py
uv run python scenarios/scenario17_horizon_role_gap/run.py
```

> 注意: FreshRetailNet-50K の列名や split 名が将来変更された場合は、各スクリプト内の feature 定義を合わせて修正してください。
