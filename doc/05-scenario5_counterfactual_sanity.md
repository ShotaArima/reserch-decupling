# Scenario 5: Counterfactual Sanity Check

## このシナリオの問い
**「global/local を入れ替えたときに、出力が意味ある方向に変わるか？」** を確認する sanity check です。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を先に参照してください。

## 1サンプルの具体化（Scenario 5）
- 入力 `x_i`: `sale_amount, discount, holiday_flag, activity_flag, avg_temperature`
- 潜在: `local_i`, `global_i`
- 反実仮想A: `local_i` 固定 + `global` シャッフル
- 反実仮想B: `global_i` 固定 + `local` シャッフル
- 出力: decoder で再生成した系列

## 評価
- `orig_mean_sale`
- `cf_global_swap_mean_sale`
- `cf_local_swap_mean_sale`

## このシナリオで言えること / 言えないこと
### 言えること
- 分離潜在を操作したときの感度を定性的に確認できる。
- 「共通/固有分離」の解釈補助として機能する。

### 言えないこと
- 主効果の因果推論にはならない（統計検定・制御実験が必要）。
- 平均値のみでは挙動の全体像を説明しきれない。

## 実行
```bash
uv run python scenarios/scenario5_counterfactual_sanity/run.py
```
