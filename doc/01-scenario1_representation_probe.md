# Scenario 1: Representation Probe

## このシナリオの問い
**「global/local の2分岐表現が、まずは計算的に学習可能か？」** を再構成タスクで確認します。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を先に参照してください。

## 1サンプルの具体化（Scenario 1）
- 入力 `x_i`:
  - `sale_amount`
  - `stock_hour6_22_cnt`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `precpt`
  - `avg_temperature`
  - `avg_humidity`
  - `avg_wind_level`
- 目的変数: `x_i` 自身（自己再構成）

## モデル
- `DecouplingAutoEncoder`（local encoder + global encoder + decoder）
- 損失: MSE
- 評価: `reconstruction_mse`

## このシナリオで言えること / 言えないこと
### 言えること
- 2分岐潜在で学習が発散せず進むかを確認できる。
- 後続タスク（予測・復元・反実仮想）へ接続する初期健全性を示せる。

### 言えないこと
- この段階だけで「global/local が意味的に完全分離できた」とは言えない。
- 予測性能の優位性は別シナリオで検証が必要。

## 実行
```bash
uv run python scenarios/scenario1_representation_probe/run.py
```
