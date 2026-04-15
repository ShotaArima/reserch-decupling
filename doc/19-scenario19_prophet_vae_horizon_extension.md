# Scenario 19: Prophet / VAE Decoupling の時間軸拡張比較

## 目的
既存の 14 日予測ベースラインを拡張し、**21 / 28 / 35 / 42 日**を追加して、
Prophet と VAE 系 Decoupling（Scenario2）の予測誤差の増え方を比較・可視化する。

- 対象 horizon: `14, 21, 28, 35, 42`
- 対象モデル:
  - Prophet（各 window を単変量時系列として next-step 予測）
  - Scenario2（DecouplingAutoEncoder + ForecastHead）
- 指標:
  - `valid_wape`, `valid_wpe`
  - `test_wape`, `test_wpe`

---

## 実装
実験コードは以下に配置。

- `scenarios/scenario19_prophet_vae_horizon_extension/run.py`

このスクリプトは horizon ごとに以下を実行する。

1. window 作成・split・正規化
2. Prophet 推論
3. Scenario2 学習・推論
4. 指標計算・保存
5. 可視化（horizon x error）

---

## 実行方法
```bash
uv run python scenarios/scenario19_prophet_vae_horizon_extension/run.py
```

---

## 出力先
`scenarios/scenario19_prophet_vae_horizon_extension/output/` に以下を保存。

- `horizon_extension_results.csv`
- `horizon_extension_valid_wape.png`
- `horizon_extension_test_wape.png`
- `h14_scenario2_train_loss.png`
- `h21_scenario2_train_loss.png`
- `h28_scenario2_train_loss.png`
- `h35_scenario2_train_loss.png`
- `h42_scenario2_train_loss.png`

---

## ログ方針
長時間実行を想定し、タイムスタンプ付きで詳細ログを出力する。

- データ読込開始/完了
- horizon 開始通知
- tensor shape と one-step pair 件数
- Prophet 実行結果（または skip 理由）
- Scenario2 の loss snapshot（start/mid/end）
- 各 artifact 保存先
- 最終サマリ（horizon x model）

---

## 期待する解釈
- horizon が長くなるほど誤差がどの程度増えるかをモデル別に確認できる。
- Prophet と Decoupling のどちらが長期化に強いかを、同じ split/指標で比較できる。
