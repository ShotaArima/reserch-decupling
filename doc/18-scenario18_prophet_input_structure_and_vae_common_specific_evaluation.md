# Scenario 18: Prophet 高精度化を狙う入力構造 × VAE(common/specific) 統合評価

## 位置づけ
Scenario18 は、これまでの Scenario9〜17 で進めてきた common/specific 分離の系譜を踏まえ、
**「Prophet に強い入力構造」と「VAE に有利な潜在遷移構造」を同一データで比較可能にする**ための設計シナリオです。

目的は単純な 1-step 予測比較ではなく、
- 入力設計（feature schema）
- モデル構造（Prophet vs Sequential VAE）
- 評価軸（点予測・区間予測・branch寄与・解釈）

を 1 本の実験計画として接続することです。

---

## 1. 実験目的

1. Prophet の精度を最大化しやすい入力データ構造を定義する。
2. 同一入力情報量の条件で、Sequential VAE（common/specific）と Prophet を比較する。
3. common/specific の役割分担が horizon とデータ条件（stockout/販促/休日）でどう変化するかを定量化する。
4. 「Prophet を超える」だけでなく、**なぜ超えたか**を latent 解析で説明可能にする。

---

## 2. 背景仮説

- **H18-1（入力構造仮説）:** Prophet は calendar・promo・価格・在庫状態を明示的に与えるほど安定改善する。
- **H18-2（構造仮説）:** `window=14 -> 1-step` の単純設定より、`lookback=28 + direct 7-step` の方が VAE の利得が出る。
- **H18-3（役割仮説）:** short horizon では specific 寄与が高く、mid horizon では common 寄与が増える。
- **H18-4（公平比較仮説）:** 同一の未来既知共変量（known future covariates）を与えた条件で、Sequential VAE は Prophet と同等以上の WAPE を示す。
- **H18-5（頑健性仮説）:** stockout subset では specific 拡張が効き、通常営業 subset では common が効く。

---

## 3. 入力データ構造（最重要）

Scenario18 では、データを **3層スキーマ**で扱う。

## 3.1 キーと時系列粒度
- 粒度: 日次
- 系列キー: `store_id × product_id`
- タイムスタンプ: `date`
- 目的変数: `sales`

## 3.2 特徴量の層分け

### A. static features（系列固定）
- `store_type`, `area_cluster`, `product_category`, `price_band`
- Prophet では系列分割または追加回帰子として利用
- VAE では embedding として encoder 初期状態に注入

### B. known-future dynamic features（予測時点で既知）
- `day_of_week`, `is_weekend`, `holiday_flag`
- `promo_planned`, `discount_rate_plan`
- `price_plan`（運用上確定している場合のみ）
- `event_flag`（販促カレンダー）

### C. observed-only dynamic features（当日時点で観測）
- `stock_hour6_22_cnt`, `hours_stock_status`
- `lagged_sales`（内生ラグ）
- `weather_observed`（未来未知なら known-future に入れない）

> 重要: Prophet と VAE の公平比較のため、未来時点に使える変数のみを「予測入力」として統一する。

## 3.3 テーブル設計

### master_series_table
- `series_id, store_id, product_id, static_*`

### daily_observation_table
- `series_id, date, sales, observed_dynamic_*`

### daily_known_future_table
- `series_id, date, known_future_*`

### modeling_view（学習用結合後）
- 上記を日付キーで結合し、欠損処理済みの学習ビューを生成

---

## 4. タスク定義

## 4.1 予測設定
- lookback: `{14, 28}`
- forecast horizon: `H=7`（direct multi-output を主設定）
- 比較用に `H=1` を補助実験として実施

## 4.2 学習サンプル
時点 `t` の入力:
- 過去 `t-L+1 ... t` の observed + known-future(過去部分)
- 未来 `t+1 ... t+H` の known-future のみ

出力:
- `sales[t+1 ... t+H]`

---

## 5. モデル群（比較対象）

## 5.1 Prophet 系

### P0: Prophet-basic
- trend + seasonality（weekly/yearly）
- 追加回帰子なし

### P1: Prophet-regressor
- known-future 特徴を追加回帰子として投入
- 休日・販促・価格計画を使用

### P2: Prophet-segmented
- カテゴリ群別または上位系列別にモデル分割
- 過学習回避のため正則化を固定

## 5.2 VAE 系

### V0: Flatten VAE（弱ベースライン）
- window flatten -> latent -> 7-step head

### V1: Sequential VAE (common/specific)
- encoder: GRU/LSTM
- latent: `z_common`, `z_specific`
- decoder: direct 7-step

### V2: Sequential VAE + latent transition（主実験）
- `z_specific(t)` の遷移ネットワークを導入
- `z_common` は遅変化正則化（temporal smoothness）
- known-future を decoder/transition に条件付け

---

## 6. common / specific の実装ルール

## 6.1 役割割当（初期設計）
- common 入力: カレンダー、価格帯、イベント、系列静的属性
- specific 入力: 在庫状態、直近売上変動、急峻な短期シグナル

## 6.2 branch ablation
- `both`
- `common_only`
- `specific_only`

## 6.3 寄与指標
誤差指標（例: WAPE）に対して
- `drop_common = metric(common_only) - metric(both)`
- `drop_specific = metric(specific_only) - metric(both)`

---

## 7. 評価設計

## 7.1 主評価
- WAPE（primary）
- MAE
- RMSE
- sMAPE

## 7.2 不確実性評価（可能なら）
- pinball loss (P50/P90)
- empirical coverage（80%, 90%）

## 7.3 subset 評価
- all
- stockout
- non-stockout
- high-promo day
- holiday window

## 7.4 horizon 分解
- `h=1..7` のステップ別誤差
- Scenario17 と接続し、horizon 依存の branch 寄与を比較

---

## 8. 実験マトリクス

最小マトリクス（推奨）:
- lookback: 2 (`14`, `28`)
- model families: 5 (`P0`, `P1`, `V0`, `V1`, `V2`)
- seeds: 3 (`42`, `52`, `62`)

合計: `2 × 5 × 3 = 30 runs`

拡張マトリクス:
- + Prophet segmented (`P2`)
- + ablation 3モード（V1/V2）

---

## 9. 実装仕様（CLI案）

## 9.1 実行例
```bash
uv run python scenarios/scenario18_prophet_vs_sequential_vae/run.py \
  --lookback 28 \
  --horizon 7 \
  --model v2_seq_vae_transition \
  --ablation-mode both \
  --seed 42
```

## 9.2 主要引数
- `--lookback {14,28}`
- `--horizon {1,7}`
- `--model {p0_prophet,p1_prophet_reg,v0_flatten_vae,v1_seq_vae,v2_seq_vae_transition}`
- `--ablation-mode {both,common_only,specific_only}`（VAEのみ）
- `--seed int`

## 9.3 出力先
`scenarios/scenario18_prophet_vs_sequential_vae/output/`
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `metrics_by_subset.csv`
- `ablation_contribution.csv`
- `prediction_samples.parquet`
- `scenario18_summary.md`

---

## 10. 判定基準（成功条件）

- **C18-1:** V2 が P1 を test WAPE で平均 1%以上改善（または同等で分散小）。
- **C18-2:** `drop_specific(h=1)` が大きく、`drop_common(h>=4)` が相対的に拡大。
- **C18-3:** stockout subset で V2 が最良、または Prophet 差が縮小せず優位。
- **C18-4:** 3 seed 中 2 seed 以上で同方向の傾向を再現。

---

## 11. 期待される解釈テンプレート

1. 「Prophet は入力構造を整えると強いが、短期状態遷移の表現で限界がある。」
2. 「Sequential VAE は known-future を条件に latent dynamics を持つため multi-step で安定。」
3. 「common/specific 分離により、精度向上と説明性（どの条件で何が効いたか）を同時に示せる。」

---

## 12. リスクと対策

- **リスク1:** Prophet の性能が系列ごとに不安定
  - 対策: グローバル集計だけでなく系列群別統計を併記

- **リスク2:** VAE が seed 依存
  - 対策: seed 固定比較 + early stopping 一貫化 + 学習曲線保存

- **リスク3:** 未来未知変数の混入によるリーク
  - 対策: known-future/observed-only のカラム検査を自動化

- **リスク4:** lookback=28 で学習コスト増
  - 対策: mixed precision / batch accumulation / プリセットエポック

---

## 13. 実行手順（運用フロー）

1. データスキーマ確定（static / known-future / observed-only）
2. リーク検査スクリプト実行
3. Prophet 系（P0, P1）を先に走らせベースライン確定
4. VAE 系（V0, V1, V2）を同一 split/seed で実行
5. V1/V2 で ablation 実施
6. subset・horizon 別に集計
7. `scenario18_summary.md` を自動生成

---

## 14. 一文まとめ

**Scenario18 は、Prophet の高精度化に効く入力構造を明示しつつ、同一条件で Sequential VAE(common/specific) の潜在遷移モデルを比較し、予測性能・寄与分解・解釈性を統合評価する実験計画である。**


---

## 15. 実験実行ステータスと結果集約（2026-04-20 時点）

このセクションでは、`scenarios/scenario18_prophet_vs_sequential_vae/output/log_*.log` を一次情報として、
「何を実験し、何が分かり、何が未達か」を 1 枚で把握できるように集約する。

### 15.1 全体進捗（実行コマンドベース vs スコア取得ベース）

- 実行ログファイル自体は **60/60 パターン分**存在。
- ただし、`[result]` 行（`valid_wape/test_wape/valid_mae/test_mae`）まで出力されたのは **16/60**。
- したがって、**「実行トリガーはほぼ完了」だが「評価可能な実験は一部のみ」**という状態。

| 区分 | 期待run数 | ログ存在 | スコア出力あり | 備考 |
|---|---:|---:|---:|---|
| 1. Prophet Models | 18 | 18 | 16 | `lb28` の `p1/p2` seed62 が欠測 |
| 2. Sequential VAE Models | 36 | 36 | 0 | 主にメモリ確保失敗/中断で結果なし |
| 3. V0 Flatten VAE | 6 | 6 | 0 | `save_learning_curve(..., output_path=...)` の TypeError |
| **合計** | **60** | **60** | **16** |  |

### 15.2 スコアが出ている実験（Prophetのみ）

#### 集約（平均は seed 平均）

| lookback | model | n(seed) | valid WAPE mean | test WAPE mean | test WAPE std |
|---:|---|---:|---:|---:|---:|
| 14 | `p0_prophet` | 3 | 1.0812 | 1.0733 | 0.0222 |
| 14 | `p1_prophet_reg` | 3 | 1.0812 | 1.0733 | 0.0222 |
| 14 | `p2_prophet_segmented` | 3 | 1.0812 | 1.0733 | 0.0222 |
| 28 | `p0_prophet` | 3 | 1.0812 | 1.0733 | 0.0222 |
| 28 | `p1_prophet_reg` | 2 | 1.1027 | 1.0756 | 0.0269 |
| 28 | `p2_prophet_segmented` | 2 | 1.1027 | 1.0756 | 0.0269 |

#### seed別（抜粋）

- `p0/p1/p2` と `lookback=14/28` の多くで同一スコア列が繰り返されており、
  例として seed42 は `valid_wape=1.0929, test_wape=1.1024`、seed52 は `valid_wape=1.1125, test_wape=1.0487`、seed62 は `valid_wape=1.0382, test_wape=1.0687` で一致。
- この一致は「モデル差/窓差が効いていない」可能性を示唆するため、実装上の切替反映を要確認。

### 15.3 スコアが出ていない実験（できなかったこと）

#### A) Sequential VAE（36 run）

- `lookback=28` 系では、`DefaultCPUAllocator: can't allocate memory ... 67737449472 bytes` がログに記録されており、
  **メモリオーバーフローで学習が停止**。
- `lookback=14` 系も多くが `[split] sample_count ...` までで終了し、`[result]` 未出力。
- 一部ログには `resource_tracker` 警告のみ、または競合マーカー文字列（`<<<<<<<`, `>>>>>>>`）が混入したものがあり、
  正常な計測ログとして扱えない。

#### B) V0 Flatten VAE（6 run）

- 全 seed / 全 lookback で `TypeError: save_learning_curve() got an unexpected keyword argument 'output_path'` により終了。
- 学習自体が完了していても、後段の保存処理エラーで実行全体が異常終了している可能性がある。

#### C) Prophet 欠測（2 run）

- `log_lb28_p1_prophet_reg_s62.log` は `zsh: command not found: $` のみ。
- `log_lb28_p2_prophet_segmented_s62.log` は空ファイル。
- したがって当該2 runは再実行が必要。

### 15.4 ここまでで分かったこと（考察）

1. **現時点での比較結論は Prophet 内の部分比較まで**
   - VAE 側に有効スコアがないため、Scenario18 の主目的である Prophet vs Sequential VAE 比較は未達。

2. **Prophet結果は「差が小さすぎる/同一値が多い」ため解釈に注意**
   - `p0/p1/p2` 間で seedごとに同値が続くため、真にモデル差を反映した結果かは未検証。
   - 入力回帰子や segmented 分岐が実際に推論へ反映されているか、run.py の分岐・特徴投入を監査すべき。

3. **ボトルネックは性能より実験基盤の安定性**
   - Sequential VAE はメモリ要件が高く、`lookback=28` で顕著に失敗。
   - V0 は保存関数引数不整合という実装不整合で全滅。
   - まず「最後まで走ってメトリクスを吐く」再現性の確立が先決。

### 15.5 次アクション（このファイルで追跡する ToDo）

- [ ] `run.py` で `save_learning_curve` の引数名を修正し、V0 6run を再実行してスコア回収。
- [ ] Sequential VAE のメモリ対策（batch縮小、系列サンプリング、mixed precision、grad accumulation）を入れて 36run を再実行。
- [ ] Prophet の `p1/p2` seed62 欠測2runを再実行。
- [ ] `p0/p1/p2` のスコア同一問題をコードレベルで検証（回帰子投入・分割学習の有効化確認）。
- [ ] 本節の表を「最新ログ再集計」で更新し、比較考察（WAPE差分、seed頑健性、horizon別）を追記。
