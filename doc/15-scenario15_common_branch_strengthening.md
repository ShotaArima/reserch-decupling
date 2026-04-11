# Scenario 15: common branch を強化すると何が改善するか

## 1. 目的

現状の `common` branch が弱く見える要因を「branch 設計そのものの限界」ではなく、
**入力特徴・表現設計の不足**として切り分ける。

本シナリオでは、`common` 側に以下を段階的に導入し、
- `common_only` の予測精度
- `common latent` の構造情報保持能力（probe）
- `both`（common + local）の統合性能

の 3 軸で改善の有無を定量検証する。

---

## 2. 検証仮説

- H1: `common` が弱いのではなく、**calendar / weather / hierarchy の与え方が不足**していた。
- H2: `common` に共有文脈（季節性・階層性・遅行統計）を入れると、`common_only` WAPE は有意に改善する。
- H3: `common latent` に calendar / hierarchy 情報がより線形可分に保持され、probe 精度が上昇する。
- H4: `both` モデルでも WAPE が改善し、`common` 強化が局所 branch の補助以上の寄与を持つ。

---

## 3. 実験条件（比較セット）

ベースラインは Scenario 6 と同様に `local_only / common_only / both` の 3 モードを採用し、
`common` 側特徴のみを操作する。

### 3.1 比較する実験アーム

- **A0 (baseline-common)**
  - 現行 `common` 入力（Scenario 6 と同等）
- **A1 (calendar-strong)**
  - `weekday`, `month`, `day_of_month`, cyclical encoding（sin/cos）を追加
- **A2 (calendar+weather-lag)**
  - A1 + weather lag / moving average 特徴を追加
- **A3 (calendar+weather+hier-embed)**
  - A2 + hierarchy embedding の次元/集約方法を見直し

### 3.2 固定条件

- 学習/評価 split: Scenario 2/6 と同一
- 予測 horizon: 既存設定（w=7, w=14 があれば両方）
- optimizer / lr / epoch / early stop 条件: 既存既定値
- seed: 3 seed（例: 42, 52, 62）

---

## 4. 実装仕様

本シナリオの実装は `scenarios/scenario15_common_branch_strengthening/run.py` を新設し、
既存の `src/scenario6_ablation.py` および `src/data.py` を流用・拡張する。

### 4.1 calendar 特徴の強化

`src/data.py` の特徴生成に以下を追加する。

- `weekday`（0-6）
- `month`（1-12）
- `day_of_month`（1-31）
- cyclical encoding
  - `weekday_sin`, `weekday_cos`（周期7）
  - `month_sin`, `month_cos`（周期12）
  - `dom_sin`, `dom_cos`（周期31）

実装上の注意:
- カテゴリ値（weekday/month/day_of_month）は embedding 用に int のまま保持。
- sin/cos は連続値として standardize 対象に含める。
- train/valid/test で同一変換を保証（fit は train のみ）。

### 4.2 weather 特徴の見直し

日次 weather に対して、アイテム/店舗共通で使える統計特徴を追加する。

追加候補（最小構成）:
- `temp_lag1`, `temp_lag7`
- `rain_lag1`, `rain_lag7`
- `temp_ma7`, `temp_ma14`
- `rain_ma7`, `rain_ma14`

実装上の注意:
- lag/MA は日付順ソート後に生成し、未来情報リークを禁止。
- 欠損（初期日）は前方利用を避け、
  - lag は 0 埋め + mask 特徴を追加、または
  - 学習対象期間を十分後ろから開始。
- まずは「0埋め + `is_weather_history_available`」を推奨。

### 4.3 hierarchy embedding の見直し

`common` 側で店舗/カテゴリなど階層情報をどの粒度で入れるかを再設計する。

検証パターン:
- `hier_dim = 4, 8, 16`
- 上位階層（例: department）と下位階層（例: item/store）の
  - concat
  - 加算（projection 後）
  - gated fusion（小規模 MLP gate）

最小実装（第一段階）:
1. `hier_dim` のみ sweep（4/8/16）
2. concat 固定

第二段階（余力があれば）:
3. concat vs add を比較
4. gated fusion を追加

---

## 5. モデル入出力定義

### 5.1 common branch 入力テンソル

- 連続特徴:
  - 既存共通特徴
  - calendar sin/cos
  - weather lag/MA
- 離散特徴:
  - weekday/month/day_of_month
  - 階層カテゴリID

### 5.2 出力

- 需要予測 `y_hat`
- 中間表現 `z_common`（probe 用に保存）

保存先:
- `scenarios/scenario15_common_branch_strengthening/outputs/`
  - `scenario15_metrics.csv`
  - `scenario15_probe_metrics.csv`
  - `scenario15_config_results.csv`
  - 学習曲線 png

---

## 6. 評価指標

### 6.1 予測性能

- 主指標: WAPE（valid/test）
- 補助: RMSE, MAE（可能なら）

比較対象:
- 各アーム A0-A3 × mode（local_only/common_only/both）

### 6.2 表現評価（probe）

`z_common` から以下を予測する簡易 probe を実装。

- calendar probe:
  - weekday（7-class）
  - month（12-class）
- hierarchy probe:
  - 上位カテゴリ class
  - 下位カテゴリ class

指標:
- classification accuracy / macro-F1

期待:
- A0 < A1 < A2/A3 の順で probe 精度向上。

### 6.3 改善判定基準（成功条件）

強い結果と見なす条件:
- `common_only` test WAPE が A0 比で **2%以上改善**（平均seed）
- calendar または hierarchy probe が **+5pt 以上改善**
- `both` test WAPE も A0 比で改善（悪化しない）

---

## 7. 実行手順

### Step 1: データ特徴追加
- `src/data.py` に calendar + weather lag/MA 生成処理を実装
- Feature schema を `src/scenario9_pipeline.py` と整合

### Step 2: common branch 入力拡張
- `src/models.py` の common encoder 入力次元を可変化
- 離散 calendar/hierarchy embedding を optional で受け取れるようにする

### Step 3: Scenario 15 ランナー作成
- `scenarios/scenario15_common_branch_strengthening/run.py` を追加
- A0-A3 のグリッド実行 + mode 切替 + seed 反復

### Step 4: probe 実装
- `src/scenario6_ablation.py` を参考に `z_common` を保存
- 簡易 logistic regression probe（別スクリプト or run.py 内）

### Step 5: レポート出力
- config ごとの valid/test WAPE 集計 csv
- best config の比較図（bar/line）
- probe 精度表

---

## 8. 解析観点

結果解釈の優先順:
1. `common_only` が改善したか（branch 単体強化の直接証拠）
2. probe が改善したか（情報保持能力の裏付け）
3. `both` が改善したか（実運用寄与）

想定される結論パターン:
- **P1: すべて改善**
  - common branch 強化方針は有効。以後は feature 設計を本線化。
- **P2: probe は改善、WAPE は横ばい**
  - 表現は改善したが decoder/融合がボトルネック。
- **P3: common_only のみ改善**
  - local と役割が競合。融合設計（gate/正則化）の再検討が必要。

---

## 9. リスクと対策

- リスク: 特徴増加で過学習
  - 対策: dropout / weight decay を固定 sweep
- リスク: weather lag の欠損処理でノイズ増
  - 対策: availability mask を必須追加
- リスク: 比較軸増えすぎで計算量過大
  - 対策: 第一段階は A0-A3 + hier_dim sweep のみ

---

## 10. 期待アウトカム

本シナリオにより、以下を明確化する。

- `common` branch は「弱い構造」ではなく、
  **共有文脈（calendar / weather 履歴 / hierarchy）投入で強化可能**か。
- 改善する場合、どの要素が支配的か
  （calendar 主因か、weather 履歴か、hierarchy embedding か）。
- 次シナリオで優先すべき設計
  （feature 投入 vs 融合アーキテクチャ改修）。

これにより、common branch の改善方向を実装レベルで決定できる。

---

## 実験結果（記録済みログ）

- ログ:
  - [output.log](../scenarios/scenario15_common_branch_strengthening/output/output.log)
  - [fixed-output.log](../scenarios/scenario15_common_branch_strengthening/output/fixed-output.log)
- 図:
  - [scenario15_valid_wape.png](../scenarios/scenario15_common_branch_strengthening/output/scenario15_valid_wape.png)
  - [scenario15_test_wape.png](../scenarios/scenario15_common_branch_strengthening/output/scenario15_test_wape.png)
- 結果メモ（`fixed-output.log`）:
  - A3（seed42, hdim=8）で `both` test WAPE=`0.4193`、`common_only` test WAPE=`0.6489`
  - probe（test）では `month accuracy=0.4567`、`weekday accuracy=0.3074`、`hierarchy accuracy=0.2844` を確認
