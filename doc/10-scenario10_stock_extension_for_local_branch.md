# Scenario 10: Stock 状態変数の local（specific）拡張実験計画

## 位置づけ
Scenario10 は Scenario9 の後段拡張です。

- Scenario9 で common/specific 分離の妥当性を確認
- Scenario10 で stock 系特徴を **specific 側に追加**し、
  local branch の状態表現強化が有効かを検証

本シナリオの主眼は、
**「stockout 関連の時変状態は local branch に載せるのが妥当か」**の確認です。

---

## 1. 実験目的

1. stock 系特徴の追加が one-step 予測を改善するか確認する。
2. 改善が出る場合、その寄与が stockout 条件で強いかを確認する。
3. common/specific の役割差を維持したまま local 表現を強化できるか確認する。

---

## 2. 追加特徴量（Scenario9 との差分）

specific encoder に以下を追加する。

- `stock_hour6_22_cnt`
- `hours_stock_status`

### 追加方針
- common 側には入れない。
- stock は時点依存の供給状態であり、local（specific）として扱う。

---

## 3. 実験系列

## Exp-3a: stock 追加（主比較）
- ベース: Scenario9 Exp-1
- 変更: specific 側へ stock 系 2 特徴を追加

## Exp-3b: stock 追加 + ablation
- both
- common-only
- specific-only

## Exp-3c: 入れ替え対照（必要時）
- stock 系を common 側へ置く対照を任意実施

> Exp-3c は「stock は local に置くべき」という設計仮説の反証実験として扱う。

---

## 4. 仮説

- **H10-1:** Exp-3a は Scenario9 Exp-1 比で性能悪化しない。
- **H10-2:** stockout subset では Exp-3a が改善しやすい。
- **H10-3:** specific latent の probe で stock 状態関連タスクが向上する。
- **H10-4:** stock を common に移す対照（Exp-3c）より、specific に置く方が自然な役割差を示す。

---

## 5. 評価設計

## 5.1 主評価（Scenario9 と同一）
- WAPE
- WPE
- MAE

## 5.2 subset 評価（推奨）
- all
- stockout
- non-stockout

> subset 定義は Scenario8 のルール（予測対象時点 mask）に合わせる。

## 5.3 追加診断
- stock 関連特徴の permutation importance（簡易）
- specific latent の stock 判別 probe

---

## 6. 入力テンソル差分

Scenario9 の `B x W x D_specific` に対して、
stock 系次元を加え `D_specific + D_stock` とする。

- `stock_hour6_22_cnt`: 連続値（正規化）
- `hours_stock_status`: 二値/カテゴリ（符号化方式を固定）

### 実装上の注意
- 欠損時は sentinel か mask で統一。
- 推論時に取得不能な特徴は使わない。

---

## 7. 成功条件

### 最低条件
- all 指標で Scenario9 Exp-1 から大幅悪化しない。

### 望ましい条件
- stockout subset で改善。
- specific-only の相対性能が改善。
- specific probe で stock 関連タスクのスコア向上。

### 強い条件
- Exp-3a が Exp-3c より解釈一貫性・性能の双方で優位。

---

## 8. 失敗時の読み方

- stockout でも改善なし:
  - stock 特徴の品質/定義を再確認
  - 容量不足（encoder hidden 次元）を確認
  - target を one-step 以外に拡張して遅延効果を確認

- all だけ改善して subset で不明瞭:
  - 偶然の可能性を排除（multi-seed）
  - 層別（カテゴリ/店舗）で再集計

---

## 9. 実行順

1. Scenario9 Exp-1 を固定 baseline として再利用
2. Exp-3a 実施
3. all/stockout/non-stockout で集計
4. Exp-3b ablation
5. 必要時 Exp-3c
6. 結果を Scenario9 と統合して解釈

---

## 10. 期待アウトプット

- `Scenario9 vs Scenario10` 比較表（主指標 + subset）
- stock 関連 probe 結果
- 「stock は local に置くべきか」の設計結論

---

## 一文まとめ
**Scenario10 は、Scenario9 の分離設計を維持したまま stock 系状態変数を specific 側へ追加し、local branch の状態表現強化と stockout 条件での有効性を検証する拡張実験である。**
