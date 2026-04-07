# Scenario 11: Latent Probe Task（common / specific の情報分離検証）

## 位置づけ
本シナリオは、Scenario9（common / specific feature assignment）の次段として、
**latent の「名前」ではなく「中身（予測可能な情報）」を probe で直接検証**する実験計画です。

狙いは次の 2 点です。

- `z_common` が **構造・文脈・長期水準**を持つことを定量的に示す。
- `z_specific` が **短期状態・施策・変動**を持つことを定量的に示す。

> 共通定義（split、評価単位など）は `doc/00-experiment_problem_setting.md` に準拠。  
> common / specific の分担前提は `doc/09-scenario9_common_specific_feature_assignment.md` を継承する。

---

## 1. 実験目的

1. **common latent の情報内容を検証する**
   - city / store / category / product などの静的・構造属性
   - 曜日 / 祝日 / 月などのカレンダー文脈
   - 長期平均売上帯（long-term level）

2. **specific latent の情報内容を検証する**
   - 次時点売上の増減方向
   - 直近変動幅
   - discount / activity / stock 状態

3. **分離の強さを “交差 probe” で示す**
   - common では強いが specific では弱いタスク
   - specific では強いが common では弱いタスク

---

## 2. 検証仮説

- **H1（common 優位）**: `z_common` は store/category/city/product、calendar、長期水準の予測で `z_specific` を上回る。
- **H2（specific 優位）**: `z_specific` は discount/activity/stock、増減方向、直近変動幅の予測で `z_common` を上回る。
- **H3（分離成立）**: それぞれの優位差が一貫して観測される（単発ではなく複数タスクで再現）。

---

## 3. Probe タスク定義

### 3.1 common latent probe（`z_common -> target`）

#### A. 構造属性分類
- `city_id`（多クラス）
- `store_id`（多クラス）
- `category_id`（多クラス）
- `product_id`（多クラス）

#### B. カレンダー識別
- `day_of_week`（7 クラス）
- `is_holiday`（2 値）
- `month`（12 クラス）

#### C. 長期平均売上帯分類
- ラベル: `long_term_sales_band`（例: Q1/Q2/Q3/Q4 の 4 クラス）
- 定義: 学習 split 内で SKU-store の長期平均売上を算出し、分位点で帯を切る。

### 3.2 specific latent probe（`z_specific -> target`）

#### D. 次時点増減方向
- `next_delta_sign = sign(y_{t+1} - y_t)`
- 実装は 3 クラス（down / flat / up）を推奨
  - `|delta| < eps` を `flat`

#### E. 直近変動幅
- `recent_volatility_band`（例: 3 クラス）
- 定義例: 直近 `k` ステップの絶対差分平均を分位点で low/mid/high に離散化

#### F. 施策・状態フラグ
- `is_discount`（2 値）
- `is_activity`（2 値）
- `is_stockout`（2 値）

---

## 4. データセット生成仕様（実装方針）

## 4.1 サンプル粒度
- 1 サンプル = 既存 one-step forecast と同じ `(entity, time t)`
- 特徴量は凍結済みモデルから抽出した `z_common(t)`, `z_specific(t)`

## 4.2 split
- `train / valid / test` は既存 split を完全再利用
- probe 学習は `train`、ハイパラ調整は `valid`、最終報告は `test`

## 4.3 ラベル漏洩防止
- `long_term_sales_band` や分位点閾値は **train split のみ**で決定
- `recent_volatility_band` の閾値も train で固定し valid/test に適用

## 4.4 保存形式
以下 2 つの中間成果物を parquet/csv で保存:

1. `latent_probe_features_{split}.parquet`
   - `sample_id`, `z_common[*]`, `z_specific[*]`, `entity_id`, `time_index`
2. `latent_probe_labels_{split}.parquet`
   - 各 probe タスクの教師ラベル

---

## 5. Probe モデル仕様（軽量・再現重視）

## 5.1 第一選択（統一）
- 線形 probe を基本とする。
  - 多クラス/2 値: Logistic Regression（L2 正則化）
  - 連続値タスクを置く場合: Ridge Regression

## 5.2 代替（補助確認）
- 1-hidden-layer MLP probe（小容量）
- 目的: 「線形では見えないが latent には含まれる」ケースの参考確認
- 主結論は線形 probe ベースで述べる。

## 5.3 入力条件
各タスクについて下記 4 条件を同一設定で学習:

1. `z_common` のみ
2. `z_specific` のみ
3. `concat(z_common, z_specific)`
4. ランダム同次元ベクトル（chance baseline）

---

## 6. 評価指標

### 分類タスク
- Macro-F1（主指標）
- Accuracy（副指標）
- 2 値不均衡タスク（discount/stock/activity）は PR-AUC 併記推奨

### 帯分類・方向分類
- Macro-F1 を主指標化（クラス不均衡耐性のため）

### 統計的安定性
- seed を変えて複数回（例: 5 seeds）実行し、平均±標準偏差を報告

---

## 7. 実験手順（実装タスク）

### Step 0: latent 抽出
- 学習済み decoupling model を eval モードで固定
- `train/valid/test` 全サンプルで `z_common`, `z_specific` を抽出して保存

### Step 1: label 生成
- 既存メタ情報から `city/store/category/product`, calendar ラベルを生成
- 売上系列から `next_delta_sign`, `recent_volatility_band`, `long_term_sales_band` を生成
- `discount/activity/stock` は既存フラグを再利用

### Step 2: probe 学習
- タスクごとに 4 入力条件（common/specific/concat/random）で学習
- valid で正則化係数などを選択し、test で固定評価

### Step 3: 集計
- タスク × 入力条件 のスコア表を作成
- `gap_common_specific = score(common) - score(specific)` を算出

### Step 4: 交差妥当性チェック
- common 優位タスク群と specific 優位タスク群で符号が一貫するか確認
- 失敗ケース（期待と逆）を個別にコメント

---

## 8. 出力フォーマット

### 表1: タスク別 probe 結果（主表）

| task | metric | common | specific | concat | random | gap(c-s) |
|---|---:|---:|---:|---:|---:|---:|
| city_id | Macro-F1 | ... | ... | ... | ... | ... |
| store_id | Macro-F1 | ... | ... | ... | ... | ... |
| category_id | Macro-F1 | ... | ... | ... | ... | ... |
| product_id | Macro-F1 | ... | ... | ... | ... | ... |
| day_of_week | Macro-F1 | ... | ... | ... | ... | ... |
| is_holiday | Macro-F1 | ... | ... | ... | ... | ... |
| month | Macro-F1 | ... | ... | ... | ... | ... |
| long_term_sales_band | Macro-F1 | ... | ... | ... | ... | ... |
| next_delta_sign | Macro-F1 | ... | ... | ... | ... | ... |
| recent_volatility_band | Macro-F1 | ... | ... | ... | ... | ... |
| is_discount | Macro-F1 | ... | ... | ... | ... | ... |
| is_activity | Macro-F1 | ... | ... | ... | ... | ... |
| is_stockout | Macro-F1 | ... | ... | ... | ... | ... |

### 表2: グループ集計（要約）

| group | tasks | avg gap(c-s) | 判定 |
|---|---|---:|---|
| 構造・文脈・長期水準 | city/store/category/product/dow/holiday/month/long-term | ... | common優位/非優位 |
| 短期状態・施策・変動 | delta sign/volatility/discount/activity/stock | ... | specific優位/非優位 |

### 図（任意）
- タスクごとの `common vs specific` 散布図（対角線つき）

---

## 9. 判定基準（Scenario11 の成功条件）

## 9.1 強い結果（理想）
- common 系タスクの大半で `common > specific`
- specific 系タスクの大半で `specific > common`
- random は全タスクで明確に下回る

この場合、以下を強く主張できる。

- **common latent は構造・文脈・長期水準を保持している**
- **specific latent は短期状態・施策・変動を保持している**
- よって latent 分離は「命名」ではなく「情報的」に成立している

## 9.2 中程度の結果
- 一部タスクで交差するが、グループ平均で期待方向
- 解釈時は「部分的分離」として慎重に記述

## 9.3 弱い結果
- 多くのタスクで差が小さい / 逆転
- 学習 objective、capacity、supervision 設計の見直し候補として記録

---

## 10. 実装ファイル計画（初版）

- `doc/11-scenario11_latent_probe_plan.md`（本計画）
- `scenarios/scenario11_latent_probe/`
  - `run.py`（end-to-end 実行）
  - `build_probe_dataset.py`（latent・label 作成）
  - `train_probe.py`（probe 学習・評価）
  - `report.py`（表出力）
  - `config.yaml`（タスク定義、閾値、seed）

---

## 11. まず着手する最小実装（MVP）

1. タスクを 6 個に絞って先行実装
   - common側: `store_id`, `category_id`, `day_of_week`
   - specific側: `next_delta_sign`, `is_discount`, `is_stockout`
2. 線形 probe のみ実装
3. 4 入力条件（common/specific/concat/random）を必須化
4. test の Macro-F1 表を 1 枚出す

この MVP で分離傾向が確認できたら、Scenario11 フル版（全 13 タスク）へ拡張する。
