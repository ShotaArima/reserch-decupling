# Scenario 3: Latent Demand Recovery with Stockout Mask

## このシナリオの問い
**「stockout 区間の扱いを変えることで、需要復元タスクの学習ループを作れるか？」** を確認します。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を先に参照してください。

## 1サンプルの具体化（Scenario 3）
- 入力 `x_i`:
  - `hours_sale`
  - `hours_stock_status`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `precpt`
  - `avg_temperature`
  - `avg_humidity`
  - `avg_wind_level`
- マスク `m_i`: `hours_stock_status > 0`
- 目的変数: `hours_sale` を含む入力再構成（マスクで重み調整）

## モデル
- `DecouplingAutoEncoder`
- 損失: 重み付き MSE
  - 非マスク: `1.0 * loss`
  - マスク: `0.1 * loss`
- 評価: WAPE, WPE（`hours_sale` 再構成）

## このシナリオで言えること / 言えないこと
### 言えること
- stockout 情報を利用した recovery 学習の最小実装が可能。
- Scenario 4 の Stage 1 設計の前提を作れる。

### 言えないこと
- 潜在需要の「真値」を復元できたとは断定できない。
- mask重み（0.1）が最適である根拠は未検証。

## 実行
```bash
uv run python scenarios/scenario3_latent_demand_recovery/run.py
```
