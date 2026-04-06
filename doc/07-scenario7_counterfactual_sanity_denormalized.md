# Scenario 7: Counterfactual Sanity Check（denormalize 後）

## 位置づけ
本シナリオは、Scenario 5（counterfactual sanity check）を
**継続すべきか / 一旦切るべきかを判断するための検証**です。

Scenario 5 の現状結果では、
- global swap と local swap がほぼ同じ
- mean sale が負値で、解釈しづらい

という課題があるため、**decode 出力を元スケールに戻した上で再評価**します。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を参照してください。

---

## 目的
Scenario 5 の counterfactual 実験結果が読めない原因が、
**単なる正規化スケール由来なのか**を切り分ける。

最終的に、Scenario 5 を今後の研究ラインとして残すかどうかを判断する。

---

## 問い
1. これは単に正規化スケールの問題か？
2. 元スケール（denormalize 後）に戻すと差が見えるのか？
3. global / local の swap は本当に意味を持つのか？

---

## やること
1. decoder 出力（original / global swap / local swap）を denormalize する
2. 3条件を同一サンプル集合で比較する
3. 平均値だけでなく、以下を追加で確認する
   - 差分分布（histogram）
   - 数系列可視化（時系列プロット）

---

## 実験条件（最小構成）
- latent 生成・swap 手順は Scenario 5 と同一
- 比較対象は以下の 3 条件
  - `original`
  - `cf_global_swap`
  - `cf_local_swap`
- 評価は **正規化空間ではなく denormalize 後**で行う
- 平均値は sales の実スケールで集計する

### denormalize の原則
- 訓練時に適用したスケーラ（例: mean/std, min/max）を利用して逆変換する
- split 間で統計が混ざらないよう、学習時と同じ統計を使う
- 逆変換対象は sales 系列（比較対象系列）に統一する

---

## 評価指標
### 1) denormalize 後の平均
- `orig_mean_sale_denorm`
- `cf_global_swap_mean_sale_denorm`
- `cf_local_swap_mean_sale_denorm`

### 2) 差分ヒストグラム
- `cf_global_swap - original`
- `cf_local_swap - original`
- 必要に応じて `cf_global_swap - cf_local_swap`

### 3) サンプル系列可視化
- 同一サンプル上で 3 系列を重ねて表示
- 少数サンプル（例: 10〜30）を固定抽出して目視比較

---

## 成功条件
- global swap と local swap の差が、denormalize 後に少なくとも可視化上で確認できる
- original と swap の差が、ノイズではなく解釈可能な方向性を持つ

---

## 想定される解釈
### 差が出る場合
- counterfactual 実験は「今後の施策比較の芽」として残せる
- 今回は sanity check 止まりだが、研究方向として継続価値がある

### 差が出ない場合
- 現時点では Scenario 5 系列は切る
- 「役割差を counterfactual で読む段階にない」と整理する
- 次の改善対象（表現分離強化、損失設計、正則化）を優先する

---

## 実行・出力（運用案）
```bash
uv run scenarios/scenario7_counterfactual_sanity_denormalized/run.py 
```
現行の `scenarios/scenario5_counterfactual_sanity/run.py` を拡張し、
以下を出力すると運用しやすい。

- denormalize 後の集計 CSV
- 差分ヒストグラム画像
- サンプル系列可視化画像

> 実装パスは Scenario 5 を再利用してよい。文書上は Scenario 7 として管理し、
> 継続/停止判断を明確に残す。
