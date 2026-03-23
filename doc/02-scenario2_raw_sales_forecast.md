# Scenario 2: Raw Sales Forecast

## このシナリオの問い
**「需要復元を挟まずに、raw sales を直接予測したときの基準性能はどの程度か？」** を確認するベースライン実験です。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を先に参照してください。

## 1サンプルの具体化（Scenario 2）
- 入力 `x_i`: `sale_amount, discount, holiday_flag, activity_flag`（正規化後）
- 目的変数 `y_i`: `sale_amount` の `HORIZON=7` 先シフト値
- 実装上の対応: `x = arr[:-7]`, `y = sale_amount[7:]`

## モデル
- 表現学習: `DecouplingAutoEncoder`
- 予測ヘッド: `ForecastHead(horizon=1)`
- 損失: L1Loss
- 評価: WAPE, WPE

## このシナリオで言えること / 言えないこと
### 言えること
- 分離潜在（local/global）を通しても、予測器が学習できるかの初期確認ができる。
- Scenario 4 と比較するための **単段ベースライン** になる。

### 言えないこと
- 「7日マルチホライズン予測ができた」とはまだ言えない（現実装は1出力）。
- 厳密な時系列窓を使うモデル比較は未実施。

## Scenario 4 との違い
- Scenario 2 は **raw直予測**（単段）。
- Scenario 4 は **Recovery → Forecast**（二段）。
- したがって S2 は「簡潔・軽量」、S4 は「設計仮説を検証しやすい」が主な違いです。

## 実行
```bash
uv run python scenarios/scenario2_raw_sales_forecast/run.py
```
