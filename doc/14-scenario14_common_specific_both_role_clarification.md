# Scenario 14: common-only / specific-only / both の役割差明確化 実験計画書

## 位置づけ
Scenario14 は、既存シナリオで観測されている **both > 単独** という事実を、
「なぜ単独だと落ちるのか」まで説明可能にするための診断シナリオです。

本シナリオでは、以下 3 条件の差分を実装レベルで比較します。

- `common_only`
- `specific_only`
- `both`

主眼は、単なるスコア比較ではなく、**誤差の質（bias / 変動追従 / 残差構造）**の分解です。

---

## 1. 目的

### 主目的
- `both > 単独` の背景を、指標・可視化・残差分析で説明可能にする。

### 副目的
- `common_only` と `specific_only` の弱点を明確化し、branch 分離の設計妥当性を強化する。
- 後続シナリオでの feature assignment 改善・loss 設計改善の判断材料を作る。

---

## 2. 強く示したい主張（このシナリオで検証）

1. `specific_only` は短期変動追従に強いが、水準復元が弱い。
2. `common_only` は水準・共有パターンは保持するが、短期状態変化に弱い。
3. `both` は両者の欠点を補完し、bias と absolute error を同時改善する。

---

## 3. 検証仮説

- **H14-1（bias 仮説）**: `common_only` は平均的に underpredict（負方向 bias）しやすい。
- **H14-2（追従仮説）**: `specific_only` は局所的変動には追従するが、予測の分散が過大化し不安定。
- **H14-3（補完仮説）**: `both` は WAPE/WPE/MAE の全体改善に加え、残差平均と残差分散の双方を改善。

---

## 4. 実験条件（固定）

- データ split: 既存シナリオ（Scenario9/10 と同一）
- 予測設定: one-step forecast（`sale_amount`）
- 学習設定: optimizer/lr/epoch/batch/seed を 3 条件で完全統一
- 入力特徴:
  - `common_only`: common branch のみ有効（specific latent をゼロ化または detach）
  - `specific_only`: specific branch のみ有効（common latent をゼロ化または detach）
  - `both`: 両 branch 有効

> 注意: 「モデル容量差」が混ざらないよう、encoder/head のパラメータ総量は可能な限り一致させる。

---

## 5. 実装スコープ

## 5.1 追加/修正対象（推奨）

- `scenarios/scenario14_role_clarification/` を新設
  - `run.py`（本シナリオ実行エントリ）
  - `outputs/`（CSV/PNG 出力先）
- `src/metrics.py`
  - 既存指標再利用（WAPE/WPE/MAE）
  - bias 指標（`mean_error`）を追加してもよい
- `src/plotting.py`
  - 予測系列比較プロット
  - 残差ヒストグラム / 残差 vs 真値散布図

## 5.2 実装ポリシー

- 既存 pipeline の再利用を優先（新規実装は最小限）。
- シナリオ実行で生成される artifact 名は固定化し、再実行で上書き可能にする。
- seed は最低 3 つ（例: 42/52/62）で平均を出す。

---

## 6. 評価設計

## 6.1 主指標（必須）

- WAPE
- WPE
- MAE

## 6.2 診断指標（必須）

- Mean Error（bias）
- 残差標準偏差（`std(residual)`）
- 変動追従性: `corr(diff(y_true), diff(y_pred))`

## 6.3 集計単位

- overall
- 推奨 subset:
  - high volatility subset（短期変動大）
  - low volatility subset（短期変動小）
  - 任意で stockout / non-stockout

---

## 7. 可視化要件（必須）

## 7.1 予測事例プロット

最低 6 系列を選定（各 2 系列 × 以下 3 タイプ）:

1. 水準ギャップが目立つ系列
2. 短期スパイクが多い系列
3. 比較的安定な系列

各系列で `y_true`, `common_only`, `specific_only`, `both` を同一グラフで表示。

## 7.2 残差可視化

- 条件別残差ヒストグラム
- 残差の箱ひげ図（条件比較）
- 残差 vs 真値（または予測値）散布図

---

## 8. 出力成果物（ファイル固定）

`scenarios/scenario14_role_clarification/outputs/` に以下を出力する。

- `scenario14_metrics_overall.csv`
  - columns: `condition, seed, wape, wpe, mae, mean_error, residual_std, diff_corr`
- `scenario14_metrics_summary.csv`
  - seed 平均・標準偏差
- `scenario14_subset_metrics.csv`
  - subset ごとの同指標
- `scenario14_predictions_sample.csv`
  - 可視化対象系列の `y_true/y_pred` 長形式データ
- `scenario14_plot_series_examples.png`
- `scenario14_plot_residual_hist.png`
- `scenario14_plot_residual_box.png`
- `scenario14_plot_residual_scatter.png`
- `scenario14_interpretation_notes.md`

---

## 9. 判定基準

### 最低条件
- `both` が overall の WAPE/WPE/MAE で単独条件より悪化しない。

### 望ましい条件
- `common_only` で負 bias（underpredict）が一貫して観測される。
- `specific_only` で `diff_corr` は高いが `residual_std` が悪化する。
- `both` で bias（|mean_error|）と MAE の両方が改善する。

### 強い条件（狙い）
- 上記傾向が multi-seed 平均でも維持される。
- subset 別でも同様の役割差が再現される。

---

## 10. 実行手順（運用レベル）

1. `both` を baseline として学習・推論。
2. 同設定で `common_only` / `specific_only` を実行。
3. 指標 CSV を統合し summary を生成。
4. 可視化対象系列を抽出し、比較プロット作成。
5. 残差可視化を作成。
6. seed 平均結果で仮説 H14-1〜H14-3 を判定。
7. `scenario14_interpretation_notes.md` に「言えること/言えないこと」を明記。

---

## 11. 失敗時の読み方（トラブルシュート）

- 差が出ない場合:
  - branch 入力分離が弱い可能性（実際に同質入力になっていないか確認）
  - head 容量過大で branch 差を吸収している可能性
- `specific_only` が過剰に悪い場合:
  - 水準情報（階層・季節）不足の影響が過大
  - 正規化/逆正規化不整合を確認
- `common_only` が想定より良い場合:
  - データが低頻度変動中心で short-term シグナルが弱い可能性

---

## 12. このシナリオ完了後に言えること

本シナリオが成功すると、次を明示できます。

- なぜ単独 branch が落ちるのか（不足情報の種類）
- `both` の改善が「単なる平均化」ではなく「役割補完」であること
- 今後の改善軸（common の bias 補正 / specific の安定化）の優先順位

---

## 一文まとめ
**Scenario14 は、common-only / specific-only / both を同一条件で比較し、WAPE/WPE/MAE・予測可視化・残差分析の三点セットで「両方必要」の中身（不足情報と補完関係）を実装レベルで説明する診断シナリオである。**

---

## 実験結果（記録済みログ）

- ログ:
  - [output.log](../scenarios/scenario14_role_clarification/output/output.log)
- 出力:
  - [scenario14_metrics_summary.csv](../scenarios/scenario14_role_clarification/output/scenario14_metrics_summary.csv)
  - [scenario14_metrics_overall.csv](../scenarios/scenario14_role_clarification/output/scenario14_metrics_overall.csv)
  - [scenario14_interpretation_notes.md](../scenarios/scenario14_role_clarification/output/scenario14_interpretation_notes.md)
- seed 平均の主結果（`scenario14_metrics_summary.csv`）:
  - `both`: wape_mean=`0.424861`
  - `common_only`: wape_mean=`0.426759`
  - `specific_only`: wape_mean=`0.428800`
