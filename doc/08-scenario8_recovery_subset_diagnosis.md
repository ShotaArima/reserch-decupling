# Scenario 8: Recovery Subset Diagnosis（A3）

## 位置づけ
本シナリオは、これまでの実験で未完となっている **A3（recovery の役割差検証）** を完了させるための切り分け実験です。  
主眼は「Scenario4 が全体で勝つか」ではなく、**stockout 区間で recovery block が機能しているか**を確認することです。

- 既存結果では、Scenario4 は overall WAPE で Scenario2 よりわずかに弱い傾向がある。
- ただしこの差だけでは「recovery は無意味」とは言えない。
- stockout / non-stockout に分けて見ることで、Scenario4 の役割を明確化する。

> 共通定義（サンプル単位、split、評価の基本）は `doc/00-experiment_problem_setting.md` を参照。  
> raw forecast の基準系は `doc/02-scenario2_raw_sales_forecast.md`、two-stage は `doc/04-scenario4_two_stage_pipeline.md`、関連アブレーションは `doc/06-scenario6_local_global_ablation.md`、denormalize の運用注意は `doc/07-scenario7_counterfactual_sanity_denormalized.md` を参照。

---

## 1. 実験目的
A3（Scenario8）の目的は次の 3 点です。

1. **Scenario4 がどの区間で有効かを確認する**  
   overall 指標だけでは見えない有効領域（特に stockout）を検証する。

2. **raw sales forecast と recovery-based forecast の違いを明確化する**  
   Scenario2（observed sales 直接予測）と Scenario4（recovery 経由）の役割差を切り分ける。

3. **Stage1 recovery が本当に recovery として機能しているかを確認する**  
   単なるノイズ源ではなく、供給制約が強い区間で寄与するかを確認する。

---

## 2. 検証仮説
### 主仮説
- **H1:** Scenario4 は overall で Scenario2 と同等以下でも、stockout 区間では Scenario2 より良い可能性がある。

### 補助仮説
- **H2:** non-stockout 区間では Scenario2 と Scenario4 の差は小さいか、Scenario2 が良い。
- **H3:** stockout 区間でも Scenario4 が優位でなければ、現状 Stage1 は recovery block として不十分である。

> 本実験は negative 結果でも価値がある。  
> 「効いていない」と分かれば、改善対象を Stage1 側に絞り込める。

---

## 3. 比較対象（最小構成）
本シナリオでは、比較対象を以下 3 本に固定します。

1. **raw baseline（推奨: FlattenMLP）**
2. **Scenario2（DecouplingAutoEncoder + ForecastHead）**
3. **Scenario4（Stage1 recovery → Stage2 forecast）**

### raw baseline の選定方針
A3 の目的は「recovery を挟む意味」の検証であるため、naive 系よりも学習ベースの代表として **FlattenMLP を raw baseline に固定**する。  
（必要に応じて LastValue は appendix 比較に回す。）

---

## 4. 公平比較の前提条件（固定項目）
以下はモデル間で必ず固定する。

- train / valid / test split
- input window size
- forecast horizon（現状 one-step）
- 正規化と denormalize の運用
- 評価対象 test set
- stockout 判定ルール（既存 mask を再利用）

**原則:** モデル以外は全て同一条件とする。

---

## 5. subset 定義
A3 の核となる定義。

1. **all:** test set の全サンプル
2. **stockout:** 既存コードの stockout mask が真のサンプル
3. **non-stockout:** `not stockout`

### 実装上の注意
mask は **予測対象時点（ラベル側）**で付与する。  
入力 window 内の stockout 有無ではなく、予測先時点の状態で subset を切る。

---

## 6. 評価対象
### 主評価
- **観測売上の one-step 予測値**（A1/A3 を地続き比較するため）

### 補助評価（任意）
- Stage1 recovery 出力そのものの subset 評価

---

## 7. 評価指標
### 主指標
- **WAPE**
  \[
  \mathrm{WAPE} = \frac{\sum |y - \hat{y}|}{\sum |y|}
  \]

- **WPE**
  \[
  \mathrm{WPE} = \frac{\sum (\hat{y} - y)}{\sum |y|}
  \]

### 補助指標
- subset 別 **MAE**

### 併記すべき母数情報
- `count`
- `target_sum`
- `stockout_ratio`

---

## 8. 実験手順
### Step 0: run 固定
- A1/A3 で共通の test split を固定
- raw baseline / Scenario2 / Scenario4 の test prediction を保存
- 正規化空間か denormalize 後かを統一

### Step 1: mask 付与
各 sample に以下のフラグを付与する。
- `is_all=True`
- `is_stockout`
- `is_non_stockout = ~is_stockout`

### Step 2: 共通評価テーブル作成
1 sample 1 行で最低限以下の列を持つ表を作る。

- `sample_id`
- `y_true`
- `y_pred_raw`
- `y_pred_s2`
- `y_pred_s4`
- `is_stockout`
- `is_non_stockout`

### Step 3: subset 集計
subset = `{all, stockout, non_stockout}` ごとに以下を算出。

- `count`
- `target_sum`
- `WAPE`
- `WPE`
- `MAE`

### Step 4: 差分表作成
主要差分:
- `Scenario4 - Scenario2`
- `Scenario4 - raw baseline`
- `Scenario2 - raw baseline`

### Step 5: 解釈テンプレート記述
subset ごとに
- 言えること
- 言えないこと
を固定フォーマットで短く記録する。

---

## 9. 出力フォーマット
### 表1: subset 別主結果

| subset       | model        | n   | target_sum | WAPE | WPE | MAE |
|--------------|--------------|-----|------------|------|-----|-----|
| all          | raw baseline | ... | ...        | ...  | ... | ... |
| all          | Scenario2    | ... | ...        | ...  | ... | ... |
| all          | Scenario4    | ... | ...        | ...  | ... | ... |
| stockout     | raw baseline | ... | ...        | ...  | ... | ... |
| stockout     | Scenario2    | ... | ...        | ...  | ... | ... |
| stockout     | Scenario4    | ... | ...        | ...  | ... | ... |
| non-stockout | raw baseline | ... | ...        | ...  | ... | ... |
| non-stockout | Scenario2    | ... | ...        | ...  | ... | ... |
| non-stockout | Scenario4    | ... | ...        | ...  | ... | ... |

### 表2: 差分表

| subset       | metric | S4 - S2 | S4 - raw | S2 - raw |
|--------------|--------|---------|----------|----------|
| all          | WAPE   | ...     | ...      | ...      |
| stockout     | WAPE   | ...     | ...      | ...      |
| non-stockout | WAPE   | ...     | ...      | ...      |
| all          | WPE    | ...     | ...      | ...      |
| stockout     | WPE    | ...     | ...      | ...      |
| non-stockout | WPE    | ...     | ...      | ...      |

---

## 10. 判定基準（定性的）
### 成功パターン A
- all: Scenario4 ≒ Scenario2
- stockout: Scenario4 改善
- non-stockout: Scenario4 同等以下

解釈: recovery は通常区間の汎用改善器ではないが、供給制約区間では有効な可能性がある。

### 成功パターン B
- stockout / non-stockout の両方で Scenario4 が一貫改善

解釈: recovery を挟む設計自体に広い有効性がある。

### 失敗パターン
- stockout でも Scenario4 が勝てない

解釈: 現状 Stage1 は recovery block として不十分であり、recovery を挟む意味は未確認。

---

## 11. 補助分析（任意）
1. **誤差ヒストグラム（subset 別）**  
   Scenario2/Scenario4 の誤差分布を重ね、tail 改善を確認。

2. **stockout 強度別評価**  
   stockout 継続長や強度 proxy で層別し、改善の出る条件を特定。

3. **Stage1 出力のケース可視化**  
   raw 観測値 / Stage1 再構成 / 最終予測を同系列で比較。

---

## 12. この実験で言えること / 言えないこと
### 言えること
- recovery block が stockout proxy 区間で機能するか
- Scenario4 が通常予測器か subset 特化か
- raw forecast と recovery-based forecast の役割差

### 言えないこと
- 真の潜在需要を厳密復元できたか
- 需要/供給/応答を完全分離できたか
- stockout mask が真の欠品を完全表現しているか

> つまり A3（Scenario8）は **latent demand 真値検証**ではなく、  
> **recovery を挟む設計の operational な有効性**を検証する実験である。

---

## 13. 実験後の分岐
### stockout 改善が見えた場合
- Stage1 の改善（損失・正則化）
- stockout case study の拡張
- recovery 専用 objective の強化

### stockout 改善が見えない場合
- stockout mask 定義の再確認
- Stage1 目的関数の再設計
- raw forecast と recovery 接続の見直し

---

## 14. 一文要約
**Scenario8（A3）は、Scenario4 が全体精度で勝つかではなく、stockout 区間で recovery block が役割を持つかを subset 別 WAPE/WPE で切り分ける本命実験である。**
