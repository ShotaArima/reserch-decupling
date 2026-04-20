---
scenario_id: S20
title: Scenario20 common-only / specific-only / both の役割差を再現性込みで診断する
status: draft
priority: A
related:
  - 002-experiment_problem_setting.md
  - 14-scenario14_common_specific_both_role_clarification.md
owner: codex
last_updated: 2026-04-20
primary_metric: WAPE
primary_baseline: both
main_claim: common と specific を分離し両 branch を併用する設計は、bias・変動追従・残差構造の補完として意味がある
---

# Scenario 20: common-only / specific-only / both の役割差を再現性込みで診断する

## 0. このシナリオで一番言いたいこと

- **何をするか**: `common_only` / `specific_only` / `both` を multi-seed・subset・残差診断まで含めて同一条件比較する。
- **何を示したいか**: 「共通表現と固有表現を分ける設計に意味がある」を、単なるスコア比較ではなく誤差の内訳で説明する。
- **主比較**: `both` vs `common_only`、`both` vs `specific_only`。
- **入力**: 既存シナリオと同一の時系列入力（特徴量・窓・split を固定）。
- **出力**: `sale_amount` の forecast（既存設定と同一 horizon）。
- **主評価**: WAPE / WPE / MAE。
- **成功条件**: `both` が seed 平均で最良、かつ bias・変動追従・残差分散の解釈が branch 役割差と整合する。
- **このシナリオの位置づけ**: Scenario14 の最小主張（both が安定）を再現性と診断粒度で確定させる。

---

## 1. 位置づけ

- 前提となる既存シナリオ: Scenario14（役割差明確化の初期検証）。
- このシナリオで新しく変えるもの:
  - seed 数を 3〜5 に拡張して再現性確認。
  - 指標を WAPE/WPE/MAE に加えて `mean_error`, `residual_std`, `corr(diff(y_true), diff(y_pred))` へ拡張。
  - representative series 可視化と high/low volatility subset 分析を追加。
- このシナリオで変えないもの:
  - データ split、前処理、基本モデル、学習ハイパーパラメータ（比較公平性に必要な範囲）。
- このシナリオが終わると何が言えるようになるか:
  - `common_only` / `specific_only` / `both` の役割差を、性能差ではなくエラー機構の差として説明できる。

---

## 2. 研究目的

### 主目的
1. `both > 単独` を seed 平均で再現し、偶然でないことを確認する。
2. 役割分担の中身（bias 寄りか、短期追従寄りか、安定性か）を定量化する。

### 副目的
1. high volatility / low volatility で branch ごとの得意不得意を分離する。
2. 後続改善（common の bias 補正、specific の安定化）の優先順位を決める材料を得る。

---

## 3. 研究質問（RQ）

- **RQ1:** `both` の優位は seed を増やしても維持されるか。
- **RQ2:** `common_only` と `specific_only` は、どの誤差特性（平均誤差・残差分散・変動追従）で差が出るか。
- **RQ3:** その差は系列の volatility（高/低）によってどう変化するか。

---

## 4. 仮説

- **H1:** `common_only` は水準寄りだが短期変化に弱く、`corr(diff(y_true), diff(y_pred))` が低下しやすい。
- **H2:** `specific_only` は短期変動追従が高い一方で不安定化し、`residual_std` が悪化しやすい。
- **H3:** `both` は補完により、WAPE/WPE/MAE と |mean_error| を同時改善する。

> 反証条件:
> - H1 が否定される: `common_only` が変動追従でも一貫して劣らない。
> - H2 が否定される: `specific_only` が追従性を保ったまま残差分散悪化を示さない。
> - H3 が否定される: `both` が単独条件に対し seed 平均で優位を示せない。

---

## 5. 30秒でわかる実験設定

| 項目 | 内容 |
|---|---|
| サンプル単位 | 既存シナリオと同一の時系列サンプル |
| 入力 | 既存設定と同一（split・窓・特徴量固定） |
| 出力 | `sale_amount` forecast |
| 比較対象 | `common_only` / `specific_only` / `both` |
| 固定条件 | 学習条件・前処理・評価対象期間 |
| 可変条件 | 有効 branch（common, specific, both）と seed |
| 主評価 | WAPE / WPE / MAE |
| 補助評価 | mean_error / residual_std / diff_corr |
| 成功条件 | `both` が seed 平均で最良、診断指標の解釈が仮説と整合 |

---

## 6. 問題設定

### 6.1 1サンプルの定義
- 既存 forecasting パイプラインにおける 1 window。
- 学習/評価の単位と split は Scenario14 と同一。

### 6.2 入力
- 使う特徴量: 既存 scenario 定義を踏襲。
- 形状: 既存モデルの入力テンソル仕様に準拠。
- 時系列窓: 既存設定を固定。
- 未来既知 / 当日観測 / static の区別: 既存定義を厳守。

### 6.3 出力
- 予測対象: `sale_amount`。
- horizon: 既存設定と同一。
- タスク種別: forecast。

### 6.4 このシナリオでの「何を当てるか」
- 絶対誤差の改善だけでなく、誤差の性質（bias・追従性・安定性）を説明できる予測を目指す。

---

## 7. 比較対象

### 7.1 主比較
- Condition A: `both`
- Condition B: `common_only`
- Condition C: `specific_only`

### 7.2 比較するモデル / 実験条件
#### Exp-20A: both
- 目的: 主ベースライン。
- 入力: common + specific 両 branch。
- 出力: `sale_amount` 予測。
- 備考: Scenario14 の最小主張の再現対象。

#### Exp-20B: common_only
- 目的: 共通表現のみの挙動確認。
- 入力: common のみ有効。
- 出力: `sale_amount` 予測。
- 備考: 水準維持と短期追従のトレードオフを診断。

#### Exp-20C: specific_only
- 目的: 固有表現のみの挙動確認。
- 入力: specific のみ有効。
- 出力: `sale_amount` 予測。
- 備考: 短期追従と安定性のトレードオフを診断。

> **主比較は Exp-20A vs Exp-20B / Exp-20C**。

---

## 8. 固定条件と可変条件

### 8.1 固定条件
- train / valid / test split
- normalization
- 欠損処理
- window size
- optimizer / lr / batch size / epochs
- 評価対象データ・評価スクリプト

### 8.2 可変条件
- branch 有効化条件（common_only / specific_only / both）
- seed（3〜5本）

### 8.3 公平比較の前提
- モデル以外は同一条件にする。
- 未来未知変数は使わない。
- leakage を防ぐ。
- 指標算出条件を統一する。

---

## 9. 評価設計

### 9.1 主評価
- WAPE
- WPE
- MAE

### 9.2 診断評価
- mean_error（bias）
- residual_std
- corr(diff(y_true), diff(y_pred))（以下 `diff_corr`）

### 9.3 サブセット評価
- high volatility subset
- low volatility subset

### 9.4 可視化
- representative series（最低 6 系列）で `y_true` と 3 条件予測を重ね描画。
- 条件別 residual histogram / boxplot / scatter。

---

## 10. 成功条件と失敗時解釈

### 10.1 成功条件（最低）
- `both` が seed 平均で WAPE/WPE/MAE の少なくとも 2 指標で最良。

### 10.2 成功条件（理想）
- `common_only`: bias か diff_corr のいずれかで弱点が明確。
- `specific_only`: diff_corr は相対的に高いが residual_std で弱点が明確。
- `both`: |mean_error| と MAE を同時に改善。

### 10.3 失敗時の意味
- 差が消える場合: branch 分離の実効性不足（入力割当や head 容量の再点検が必要）。
- `both` が優位でない場合: 補完設計の仮説再検討（融合方法・loss 設計の見直し）。

---

## 11. 実行・成果物

### 11.1 実行手順
1. 各条件を seed=3〜5 で実行。
2. overall / subset 指標を集計。
3. representative series と残差可視化を作成。
4. 仮説 H1〜H3 の採否を記録。

### 11.2 出力ファイル（想定）
- `scenario20_metrics_overall.csv`
- `scenario20_metrics_summary.csv`
- `scenario20_metrics_subset.csv`
- `scenario20_predictions_sample.csv`
- `scenario20_plot_series_examples.png`
- `scenario20_plot_residual_hist.png`
- `scenario20_plot_residual_box.png`
- `scenario20_plot_residual_scatter.png`
- `scenario20_interpretation_notes.md`

---

## 12. 現時点の既知情報（Scenario14 からの継承）

Scenario14 の seed 平均では以下が観測されている。

- `both`: 0.424861
- `common_only`: 0.426759
- `specific_only`: 0.428800

このため、**「両 branch 併用が最も安定」**という最小主張は暫定的に支持される。
Scenario20 はこの結果を、再現性と診断軸を増やして確証へ進めるための計画である。

---

## 一文まとめ

**Scenario20 は、common-only / specific-only / both の性能差を seed 再現性・誤差分解・subset 診断で再確認し、「共通表現と固有表現の分離には補完的な意味がある」という中心主張を最も直接に検証するシナリオである。**
