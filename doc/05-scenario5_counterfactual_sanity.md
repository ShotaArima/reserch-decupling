# Scenario 5: Counterfactual Sanity Check

## 目的
global/local 分離が意味を持つかを、**反実仮想生成の最小 sanity check**で確認します。

## 背景と狙い
- 分離表現の妥当性は「片方を固定してもう片方だけ変えた時の挙動」で確認しやすいです。
- 本シナリオでは定量主結果ではなく、解釈補助としての挙動確認を狙います。

## 入力
- データセット: `FreshRetailNet/FreshRetailNet-50K`
- 特徴量
  - `sale_amount`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `avg_temperature`

## モデル構成
- `DecouplingAutoEncoder` で `local`, `global` を取得
- 反実仮想 A
  - local 固定 + global シャッフル
- 反実仮想 B
  - global 固定 + local シャッフル
- decoder に再投入して系列を生成

## 出力
- `orig_mean_sale=...`
- `cf_global_swap_mean_sale=...`
- `cf_local_swap_mean_sale=...`

## 何を確認するか
- global/local の交換で出力統計がどう変化するか
- 変化方向がドメイン直観と矛盾していないか

## 実行
```bash
uv run python scenarios/scenario5_counterfactual_sanity/run.py
```

## 次の発展
- 条件付き平均、割引感応度、休日差分などのドメイン指標を追加
- 可視化図（時系列重ね描き）で定性的説明を強化
