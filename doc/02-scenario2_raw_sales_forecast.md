# Scenario 2: Raw Sales Forecast (7-day)

## 目的
Decoupling-inspired 表現を使って、**raw sales の予測器として最低限機能するか**を検証する最小実験です。

## 背景と狙い
- 表現学習モデルは解釈性だけでなく、下流予測での有効性が重要です。
- まずは daily 集約の簡易設定で 7 日先を意識した最小形（実装上は horizon=1 を rolling 的に扱う下準備）を確認します。

## 入力
- データセット: `FreshRetailNet/FreshRetailNet-50K`
- 特徴量
  - `sale_amount`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
- 目的変数
  - `sale_amount`

## モデル構成
- 表現学習部: `DecouplingAutoEncoder`
- 予測部: `ForecastHead`
- 損失: L1Loss
- 評価: `WAPE`, `WPE`

## 出力
- `WAPE=...`
- `WPE=...`

## 何を確認するか
- 単純な設定でも forecast 指標を出力できるか
- local/global 表現を下流予測に接続する実験導線が成立しているか
- 今後の比較（DLinear、MLP、global無し版）を載せる基盤があるか

## 実行
```bash
uv run python scenarios/scenario2_raw_sales_forecast/run.py
```

## 次の発展
- 真の 7-step 予測に変更
- train/valid/test の時間分割を明示
- 比較ベースライン（DLinear, RNN, MLP）を統一条件で追加
