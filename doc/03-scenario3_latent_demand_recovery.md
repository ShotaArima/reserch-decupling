# Scenario 3: Latent Demand Recovery with Stockout Mask

## 目的
stockout 情報をマスクとして活用し、**観測 sales から潜在需要を復元する方向**を検証する最小実験です。

## 背景と狙い
- stockout 区間では観測 sales が需要を過小反映する可能性があります。
- そのため、stockout 区間の損失重みを調整し、復元学習の入口を作ります。

## 入力
- データセット: `FreshRetailNet/FreshRetailNet-50K`
- 主特徴量
  - `hours_sale`
  - `hours_stock_status`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `precpt`
  - `avg_temperature`
  - `avg_humidity`
  - `avg_wind_level`

## モデル構成
- `DecouplingAutoEncoder` による再構成
- `hours_stock_status > 0` をマスクとして利用
- 重み付き再構成損失
  - 非マスク区間: 通常重み
  - マスク区間: 低重み（最小実装）
- 評価: `WAPE`, `WPE`（`hours_sale` 再構成に対して）

## 出力
- `WAPE=...`
- `WPE=...`

## 何を確認するか
- mask を使う recovery 学習ループが動作するか
- 復元タスクへの拡張可能性（MNAR simulation など）の土台になるか

## 実行
```bash
uv run python scenarios/scenario3_latent_demand_recovery/run.py
```

## 次の発展
- マスク区間での censoring 仮定をより厳密化
- controlled MNAR の人工欠損実験
- recovery 後系列の予測改善効果を scenario4 で比較
