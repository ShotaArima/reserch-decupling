# Scenario 4: Two-stage Pipeline (Recovery → Forecast)

## 目的
FreshRetailNet-50K の利用意図に沿って、**需要復元 → 将来予測**の二段階構成を最小形で確認します。

## 背景と狙い
- 実運用に近い流れとして、まず需要信号を復元してから forecasting する構成が有効です。
- 本シナリオはその研究ストーリーを最短で実装化したものです。

## Stage 1: Recovery
- 入力特徴
  - `sale_amount`
  - `hours_stock_status`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
- モデル
  - `DecouplingAutoEncoder`
- 出力
  - 再構成系列（recovered proxy）

## Stage 2: Forecast
- 入力
  - Stage 1 で得られた local/global 表現
- モデル
  - `ForecastHead`
- 損失
  - L1Loss
- 評価
  - `WAPE`

## 出力
- `stage1_recovery_mse=...`
- `stage2_wape=...`

## 何を確認するか
- 2 段階パイプラインがコード上で分離されているか
- recovery と forecasting を将来独立に改善できる構造になっているか

## 実行
```bash
uv run python scenarios/scenario4_two_stage_pipeline/run.py
```

## 次の発展
- Stage1 を hourly demand recovery に拡張
- Stage2 を 7日先 multi-horizon に変更
- raw 直接予測との比較で recovery の寄与を定量化
