# Scenario 2: Raw Sales Forecast (Experiment 1: Forecast Baseline Block)

## 目的
現在の `Scenario 2 raw sales forecast` が、
**本当に意味のある性能なのか** を判断できるようにする。

現状では raw sales forecast は成立しているが、
その値が naive baseline より良いのか、また Decoupling 風表現が役立っているのかがまだ見えにくい。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を参照してください。

## 問い
- `Scenario 2` は naive forecast より良いのか？
- 窓長 14 にする意味はあるのか？
- two-stage にする前に、そもそも素朴な predictor と比べてどうか？

## 入力
- 直近 `W=14` の window tensor（比較の主設定）
- 特徴量はモデルごとに下記で固定し、同一 split（train/valid/test）で比較する

## 出力
- 次時点の raw sales（`x_t -> y_{t+1}`）

## 比較対象（モデル仕様）

### 1) Last Value
- 予測式: `ŷ_{t+1} = sale_amount_t`
- 入力特徴量: `sale_amount` のみ（window の最終時点）
- 主なパラメータ:
  - window: `W=14`（ただし予測に使うのは最終値のみ）
  - 学習パラメータ: なし

### 2) Moving Average
- 予測式: `ŷ_{t+1} = mean(sale_amount_{t-k+1:t})`
- 入力特徴量: `sale_amount` のみ
- 主なパラメータ:
  - window: `W=14`
  - 平均区間: `k ∈ {3, 7, 14}`（valid で最良の `k` を採用）
  - 学習パラメータ: なし

### 3) Flatten + Linear
- 予測式: `flatten(window) -> Linear -> ŷ_{t+1}`
- 入力特徴量: `sale_amount, discount, holiday_flag, activity_flag`
- 主なパラメータ:
  - window: `W=14`
  - 入力次元: `14 × 4 = 56`
  - 出力次元: `1`
  - 損失: L1Loss
  - 最適化: Adam (`lr=1e-3`)
  - 学習 step: `100`

### 4) Flatten + MLP
- 予測式: `flatten(window) -> MLP(2〜3層) -> ŷ_{t+1}`
- 入力特徴量: `sale_amount, discount, holiday_flag, activity_flag`
- 主なパラメータ（初期設定）:
  - window: `W=14`
  - 入力次元: `56`
  - 隠れ層候補: `[128, 64]` または `[128, 64, 32]`
  - 活性化: ReLU
  - 出力次元: `1`
  - 損失: L1Loss
  - 最適化: Adam (`lr=1e-3`, `weight_decay=1e-5`)
  - 学習 step: `100`

### 5) Scenario 2（current raw sales forecast）
- 構成: `DecouplingAutoEncoder + ForecastHead`
- 入力特徴量: `sale_amount, discount, holiday_flag, activity_flag`
- 主なパラメータ（現行実装）:
  - window: 実装は `W ∈ {7, 14}`（本比較では `W=14` を主設定）
  - `DecouplingConfig(feature_dim=4, window_size=W)`
  - `ForecastHead(local_dim=16, global_dim=16, horizon=1)`
  - 損失: L1Loss
  - 最適化: Adam (`lr=1e-3`)
  - 学習 step: `100`

### 6) Scenario 4（current two-stage pipeline）
- 構成: Stage 1 `DecouplingAutoEncoder`（recovery）→ Stage 2 `ForecastHead`
- 入力特徴量: `sale_amount, hours_stock_status, discount, holiday_flag, activity_flag`
- 主なパラメータ（現行実装）:
  - window: 実装は `W ∈ {7, 14}`（本比較では `W=14` を主設定）
  - Stage 1:
    - `DecouplingConfig(feature_dim=5, window_size=W)`
    - 損失: MSELoss（再構成）
    - 最適化: Adam (`lr=1e-3`)
    - 学習 step: `100`
  - Stage 2:
    - `ForecastHead(local_dim=16, global_dim=16, horizon=1)`
    - 損失: L1Loss
    - 最適化: Adam (`lr=1e-3`)
    - 学習 step: `100`

## 評価指標
- valid/test `WAPE`
- valid/test `WPE`

## 成功条件
- `Scenario 2` が Last Value / Moving Average / Flatten + Linear に対して優位
- 少なくとも naive（Last Value, Moving Average）と同等以下でないこと
- `Scenario 4` が `Scenario 2` に勝てなくても、その差（改善/劣化）が定量的に見えること

## 想定される解釈
### 良い場合
- Decoupling 風構成は naive baseline を超える
- 小売時系列にも一定の表現学習の価値がある

### 悪い場合
- 直近値や単純平均で足りている
- まだ Decoupling 的な構成の優位性は弱い
- 特徴量またはタスク設計の見直しが必要

## 実行
```bash
uv run python scenarios/scenario2_raw_sales_forecast/run.py
```
