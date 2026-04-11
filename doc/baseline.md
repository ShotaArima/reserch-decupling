# Baseline 実験の構想と実行結果

## 1. 目的（構想）

本ドキュメントは、Forecast タスクに対して次の2点を明確化するための baseline 集約ノートです。

1. **Decoupling 系モデル（Scenario2/4）が、単純ベースラインより実際に良いか**
2. **改善幅がどの程度か**（WAPE/WPE ベース）

比較対象は、
- Naive（LastValue / MovingAverage）
- タブラー回帰（FlattenLinear / FlattenMLP）
- 時系列統計（Prophet）
- Decoupling（Scenario2 / Scenario4）

を同一実行ブロックで走らせて比較する構成です。

---

## 2. 実行ログ

- 実行ログ: [baselines/forecast_block/outputs/baseline.log](../baselines/forecast_block/outputs/baseline.log)
- 補助ログ: [baselines/forecast_block/outputs/output.log](../baselines/forecast_block/outputs/output.log)

ログ上で確認できる実行設定/選択:
- `MovingAverage` は `k=14` が選択
- `FlattenMLP` は `hidden_dims=[128, 64, 32]` が選択
- Prophet baseline も実行

生成物（ログ記載）:
- `flatten_linear_train_loss.png`
- `flatten_mlp_train_loss.png`
- `scenario2_train_loss.png`
- `scenario4_stage1_train_loss.png`
- `scenario4_stage2_train_loss.png`
- `forecast_baseline_results.csv`
- `valid_wape_comparison.png`
- `test_wape_comparison.png`

---

## 3. baseline.log の結果サマリ

`baseline.log` の `=== Result Summary ===` より:

| Model | valid_wape | valid_wpe | test_wape | test_wpe |
|---|---:|---:|---:|---:|
| LastValue | 0.4636 | -0.0000 | 0.4584 | 0.0000 |
| MovingAverage | 0.4636 | -0.0000 | 0.4584 | -0.0000 |
| FlattenLinear | 0.8703 | -0.8592 | 0.8696 | -0.8596 |
| FlattenMLP | 0.4287 | -0.0479 | 0.4237 | -0.0496 |
| Prophet | 0.4636 | -0.0000 | 0.4584 | 0.0000 |
| Scenario2 | 0.4287 | -0.0524 | 0.4237 | -0.0539 |
| Scenario4 | 0.4284 | -0.0536 | 0.4242 | -0.0554 |

---

## 4. 解釈メモ

- **最良 valid WAPE は Scenario4（0.4284）**。
- **最良 test WAPE は FlattenMLP / Scenario2（0.4237）** で同率、Scenario4（0.4242）が僅差で続く。
- Naive/Prophet（0.4584）と比べると、Decoupling 系および FlattenMLP は改善している。
- FlattenLinear は大きく劣後しており、単純線形 flatten の表現力不足が示唆される。

---

## 5. この baseline から次に見るべき点

- seed 複数化で順位の安定性確認
- subset（stockout/高変動）ごとの差分
- WAPE だけでなく残差分布・WPE バイアスの比較

