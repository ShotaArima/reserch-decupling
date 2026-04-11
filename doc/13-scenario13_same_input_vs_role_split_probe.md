# Scenario 13: same input では分離が起きにくいことを示す実験計画

## 0. 位置づけ

本シナリオは Scenario9 の解釈を明確化するための **検証専用シナリオ** である。特に、

- `exp0_same_input`（branch は分離しているが入力特徴は同一）
- `exp1_role_split`（branch ごとに役割に沿って入力を分離）

を比較し、**「branch を分けるだけでは latent 分離は起きにくい」**ことを、probe ベースで強く示す。

---

## 1. 目的

### 主目的
- Scenario9 の Exp-0 が持つ意味（same input 条件の限界）を、表現解析で明示する。

### 副目的
- 分離成立にはアーキテクチャの branch 分割だけでなく、**特徴量設計（role-aware input design）** が必要であることを実証する。

---

## 2. 検証したい主張

1. **same input 条件では** `z_common` / `z_specific` の保持情報が似やすい。  
2. **role split 条件では** `z_common` / `z_specific` の得意タスクが分かれやすい。  
3. よって「分離には特徴量設計が必要」という主張が、予測指標に加えて表現指標でも支持される。

---

## 3. 実験設計（比較対象）

## 3.1 対象 run
- ベース: `scenarios/scenario9_common_specific_feature_assignment/run.py`
- 比較対象 experiment:
  - `exp0_same_input`
  - `exp1_role_split`

## 3.2 固定条件
- データ split, window size, 学習 step, seed は同一。
- モデル構造（encoder/head）も同一。
- 違いは branch への入力特徴割り当てのみ。

## 3.3 追加する解析対象
- 学習済みモデルから抽出した latent:
  - `z_common`
  - `z_specific`
- split: valid/test（最終報告は test を主、valid は補助）

---

## 4. Probe 実装方針（実装レベル）

## 4.1 実装ファイル構成（新規追加）

- `src/scenario13_probe.py`（新規）
  - latent 抽出
  - probe 学習・推論
  - 指標計算
  - 類似度計算（cross-latent similarity）
- `scenarios/scenario13_same_input_vs_role_split_probe/run.py`（新規）
  - Scenario9 の実験を呼び出し（または再実行）
  - exp ごとに probe を実行
  - CSV / 図表を出力
- `scenarios/scenario13_same_input_vs_role_split_probe/output/`（新規）
  - `scenario13_probe_scores.csv`
  - `scenario13_latent_similarity.csv`
  - `scenario13_probe_gap_summary.csv`
  - `scenario13_probe_score_heatmap.png`
  - `scenario13_similarity_bar.png`

> 既存 Scenario9 の資産を使うため、まずは run 内で Scenario9 相当の学習を再実行し、次段で latent 抽出する。将来的に checkpoint 保存へ拡張可能。

## 4.2 Probe タスク定義

### A. 構造寄り（common 優位を期待）
- `city_id` multi-class classification
- `store_id` multi-class classification
- `first_category_id` multi-class classification
- `holiday_flag` binary classification

### B. 状態寄り（specific 優位を期待）
- 次時点増減方向（`sale_amount[t+1] - sale_amount[t] > 0`）binary classification
- 直近変動幅の高低（Δsale の上位 quantile 判定）binary classification
- `discount` の有無 binary classification
- `activity_flag` の有無 binary classification

## 4.3 Probe モデル
- まずは軽量で比較可能な固定 probe を採用:
  - 分類: Logistic Regression
  - 回帰が必要な場合: Ridge Regression
- 入力は latent 単体:
  - `probe(z_common)`
  - `probe(z_specific)`

## 4.4 評価指標
- 分類: Accuracy, Macro-F1, ROC-AUC（2値のみ）
- 回帰（使う場合）: R2, MAE
- 主要比較量:
  - `score_common(task)`
  - `score_specific(task)`
  - `gap(task) = score_common - score_specific`

---

## 5. 「分離の起きにくさ」を強く示すための追加指標

## 5.1 Latent 間類似度
exp ごとに `z_common` と `z_specific` の類似度を算出する。

- CKA（推奨）
- 補助: 平均 cosine similarity

期待:
- `exp0_same_input`: 類似度が高い
- `exp1_role_split`: 類似度が低下

## 5.2 Probe 分担度（Role Separation Index）
タスク集合を `T_common`, `T_specific` として

- `RSI_common = mean_{t in T_common}(score_common(t) - score_specific(t))`
- `RSI_specific = mean_{t in T_specific}(score_specific(t) - score_common(t))`
- `RSI_total = (RSI_common + RSI_specific)/2`

期待:
- `exp0_same_input`: `RSI_total` が小さい（0 近傍）
- `exp1_role_split`: `RSI_total` が正に拡大

---

## 6. 具体的な実装手順

1. `src/scenario13_probe.py` を追加し、以下 API を実装する。  
   - `extract_latents(model, splits, split) -> dict[str, np.ndarray]`  
   - `build_probe_labels(df, split_indices) -> dict[str, np.ndarray]`  
   - `run_probes(latents, labels, probe_config) -> pd.DataFrame`  
   - `compute_latent_similarity(z_common, z_specific) -> dict[str, float]`  
   - `summarize_role_separation(probe_df) -> pd.DataFrame`
2. `scenarios/scenario13.../run.py` を追加。  
   - Scenario9 と同じ data loading / feature resolution を使用。  
   - `exp0_same_input` と `exp1_role_split` のみ実行。  
   - 学習済み model から test latent 抽出。  
   - 各 task で `z_common`/`z_specific` probe を学習評価。  
   - 類似度・RSI を算出。  
   - CSV と図を保存。
3. 出力 CSV の schema を固定する。  
   - `scenario13_probe_scores.csv`: `experiment, task, latent, metric, score`  
   - `scenario13_latent_similarity.csv`: `experiment, cka, cosine_mean`  
   - `scenario13_probe_gap_summary.csv`: `experiment, rsi_common, rsi_specific, rsi_total`
4. 図表出力。  
   - task x latent の score heatmap（exp ごと）  
   - similarity / RSI の棒グラフ

---

## 7. 判定基準（強い結果）

## 7.1 強い結果の最小セット
- `exp0_same_input` で、同一 task に対する `z_common` と `z_specific` の probe 成績が近い。  
- `exp1_role_split` で、構造タスクは `z_common` 優位、状態タスクは `z_specific` 優位に分かれる。  
- `exp1_role_split` の `RSI_total > exp0_same_input`。  
- `exp1_role_split` の latent 類似度（CKA）が `exp0_same_input` より低い。

## 7.2 期待される解釈
- same input では「どちらの branch でも同じ情報を再表現」しがち。  
- role split では入力の情報源制約により latent 機能分化が進む。  
- よって、**分離には特徴量設計が必要**という主張を強く支持できる。

---

## 8. 実行コマンド案

```bash
# Scenario13 実行（例）
python scenarios/scenario13_same_input_vs_role_split_probe/run.py \
  --window-size 14 \
  --steps 120 \
  --seed 42
```

必要に応じて multi-seed 化:

```bash
python scenarios/scenario13_same_input_vs_role_split_probe/run.py --seed 42
python scenarios/scenario13_same_input_vs_role_split_probe/run.py --seed 43
python scenarios/scenario13_same_input_vs_role_split_probe/run.py --seed 44
```

---

## 9. リスクと対策

- probe の過学習: `train/valid` でハイパラ固定し test は最終評価のみ。  
- クラス不均衡: Macro-F1 と class_weight を併用。  
- seed 依存: 3 seed 以上で平均と分散を報告。  
- task 難易度差: score の絶対値だけでなく `common-specific gap` を主比較指標にする。

---

## 10. 最終アウトプット

- 実験結果表（probe score, gap, RSI, latent similarity）
- 図表（heatmap, bar chart）
- 解釈メモ（Scenario9 Exp-0 の意味の明確化）

---

## 一文まとめ

**Scenario13 は、`exp0_same_input` と `exp1_role_split` の probe 比較により、branch 分割だけでは latent 分離が起きにくく、分離には特徴量設計が必要であることを実装可能な形で実証する計画である。**
