# Scenario 17: 予測 horizon 変化による common / specific 役割差検証

## 位置づけ
Scenario17 は Scenario9/10 で定義した common / specific 分離設計を前提に、
**予測 horizon（何ステップ先を当てるか）で branch の寄与がどう変わるか**を検証するシナリオです。

狙いは、次の主張を実験的に強めることです。

- 短期（1-step）では specific（直近状態）寄与が強い
- 少し長め（3-step / 7-step）では common（構造・共有文脈）の相対寄与が増える

---

## 1. 実験目的

1. horizon を `1-step / 3-step / 7-step` に変えたときの予測性能差を比較する。
2. horizon ごとに `common-only / specific-only / both` を比較し、branch の役割差を定量化する。
3. 「役割差は horizon 依存である」という解釈可能な結論を得る。

---

## 2. 検証したい仮説

- **H17-1:** `1-step` では `specific-only` が `common-only` より高性能。
- **H17-2:** `3-step`, `7-step` と horizon を伸ばすと `common-only` の相対性能が改善（または劣化幅が小さく）する。
- **H17-3:** `both` は全 horizon で安定して最良または準最良を維持する。
- **H17-4:** `specific-only` と `common-only` の性能差は horizon に応じて縮小または逆転傾向を示す。

---

## 3. 実験条件（固定）

## 3.1 ベース設計
- 特徴量割り当ては Scenario9 Exp-1 を採用（common/specific 分離）。
- stock 拡張を入れる場合は Scenario10 Exp-3a を別系統で実施（混在させない）。

## 3.2 変更する要素
- 予測 horizon のみ変更: `h ∈ {1, 3, 7}`。

## 3.3 固定する要素
- split（train/valid/test）
- 正規化/欠損処理/カテゴリ未知値処理
- モデル容量（hidden 次元、層数）
- 最適化条件（optimizer, lr, epochs, batch size）
- seed セット

> 目的は horizon 効果の同定なので、horizon 以外は可能な限り固定する。

---

## 4. 実装レベル仕様

## 4.1 データセット生成
各サンプル時点 `t` に対して、目的変数 `y` を次で定義する。

- `1-step`: `y = sale_amount[t+1]`
- `3-step`: `y = sale_amount[t+3]`
- `7-step`: `y = sale_amount[t+7]`

### 実装要件
- 既存の window 生成関数に `forecast_horizon: int` を追加。
- valid/test の末尾で `t+h` が取れないサンプルは除外。
- 各 horizon で train/valid/test の件数をログ保存。

## 4.2 モデル実行モード
各 horizon で以下 3 モードを実行。

1. `both`（通常）
2. `common_only`（specific latent をゼロ化 or detach）
3. `specific_only`（common latent をゼロ化 or detach）

### 実装要件
- 推論時だけでなく学習時にも同一モードで訓練（推奨）。
- 最低限、評価時アブレーションを必須化（学習済み both から分岐可）。
- モード切替を CLI 引数化：`--ablation-mode both|common_only|specific_only`。

## 4.3 実行マトリクス
最小実行単位は以下。

- horizons: 3 条件（1, 3, 7）
- modes: 3 条件（both, common_only, specific_only）
- seeds: 推奨 3（例: 42, 43, 44）

合計: `3 × 3 × 3 = 27` run（最小推奨）。

---

## 5. CLI / ディレクトリ設計案

## 5.1 run コマンド例
```bash
uv run python scenarios/scenario17_horizon_role_gap/run.py \
  --forecast-horizon 1 \
  --ablation-mode both \
  --seed 42
```

## 5.2 出力先
`scenarios/scenario17_horizon_role_gap/outputs/` 配下に以下を保存。

- `metrics_h{h}_{mode}_seed{seed}.csv`
- `predictions_h{h}_{mode}_seed{seed}.csv`（必要時）
- `train_loss_h{h}_{mode}_seed{seed}.png`
- `summary_by_horizon.csv`（全 run 集約）
- `relative_contribution.csv`（寄与指標）

---

## 6. 評価指標

## 6.1 主指標
- WAPE
- WPE
- MAE

## 6.2 追加指標（推奨）
- sMAPE（スケール差補助）
- RMSE（大誤差感度確認）

## 6.3 集計
- seed 平均 ± 標準偏差
- horizon × mode のピボット表

---

## 7. 役割差の定量化（重要）

`both` を基準として、horizon ごとに寄与差を定義する。

- `drop_common(h) = metric(common_only, h) - metric(both, h)`
- `drop_specific(h) = metric(specific_only, h) - metric(both, h)`

（metric は誤差系のため小さいほど良い。値が大きいほど落差が大きい。）

### 解釈
- `drop_specific(1) > drop_common(1)` なら 1-step は specific 依存が強い。
- `drop_common(7)` が相対的に縮小、または `common_only` が改善するなら、長め horizon で common の価値が上がる。

---

## 8. 判定基準（強い結果）

以下を満たすほど主張が強い。

1. `1-step` で `specific-only` が `common-only` より有意に良い。
2. `7-step` で `common-only` の順位または相対性能が改善。
3. `both` が全 horizon で安定して上位。
4. seed を跨いで傾向が再現。

### 最低成立ライン
- 少なくとも 3 seed 中 2 seed 以上で同一傾向。

### 推奨（統計）
- horizon ごとに paired 比較（seed 対応）
- 効果量（Cohen's d など）併記

---

## 9. 可視化

最低限、以下 3 種を作成。

1. **horizon × mode の棒グラフ**（WAPE/WPE/MAE）
2. **寄与落差プロット**（`drop_common`, `drop_specific` vs horizon）
3. **順位遷移図**（mode の順位変化）

解釈を強化するため、図タイトルに horizon を明記する。

---

## 10. 実行手順（運用）

1. `run.py` に `--forecast-horizon` と `--ablation-mode` を実装。
2. 単発 dry-run（h=1, both, seed=42）で動作確認。
3. 27 run をバッチ実行。
4. `summary_by_horizon.csv` を自動生成。
5. 可視化スクリプトで図を出力。
6. Scenario9/10 と整合する解釈文を作成。

---

## 11. リスクと対策

- **リスク1:** horizon が長いほどデータ数が減る。
  - 対策: 各 horizon のサンプル数を明示し、比較時に併記。

- **リスク2:** 7-step で全モードが悪化し差が不明瞭。
  - 対策: 指標を複数化し、relative drop で比較。

- **リスク3:** seed 依存で結論がぶれる。
  - 対策: seed 数を 5 まで増やし傾向の頑健性を確認。

---

## 12. 期待アウトプット

- horizon 依存の役割差を示す定量表（CSV）
- 可視化 3 点セット
- 結論テンプレート：
  - 「短期は specific 優位、長め horizon では common の相対寄与が増加」
  - 「both は全域で安定」

---

## 一文まとめ
**Scenario17 は、予測 horizon（1/3/7-step）を操作して common/specific branch の相対寄与を比較し、役割差が時系列先読み長に依存することを定量的に示すための実装可能な実験計画である。**

---

## 実験結果（記録済みログ）

- ログ:
  - [output.log](../scenarios/scenario17_horizon_role_gap/output/output.log)
- 集計:
  - [summary_by_horizon.csv](../scenarios/scenario17_horizon_role_gap/output/summary_by_horizon.csv)
  - [relative_contribution.csv](../scenarios/scenario17_horizon_role_gap/output/relative_contribution.csv)
- 要約（seed=42）:
  - h=1: `specific_only` test WAPE=`0.4281`, `both`=`0.4351`, `common_only`=`0.6670`
  - h=3: `specific_only` test WAPE=`0.4742`, `both`=`0.4756`, `common_only`=`0.6627`
  - h=7: `specific_only` test WAPE=`0.4676`, `both`=`0.4730`, `common_only`=`0.6617`
