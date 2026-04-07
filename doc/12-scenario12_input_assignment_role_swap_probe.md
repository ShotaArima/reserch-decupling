# Scenario 12: 入力割り当て変更時の latent 役割入れ替わり検証（実装レベル計画書）

## 位置づけ
Scenario9 で定義した `exp1_role_split` / `exp2_swapped_split` を土台に、
**「branch 名（common/specific）そのもの」ではなく「どの特徴量をどちらに入力したか」で latent の役割が決まるか**を検証する。

本シナリオは、次の主張を強くするための検証である。

- 弱い主張: 2 branch 化で何らかの分離は起きる。
- 強い主張: 役割分離は入力設計により誘導される（branch 名に固定されない）。

---

## 1. 目的と検証仮説

## 1.1 目的
- `exp1_role_split` と `exp2_swapped_split` の双方に同一 probe 群を適用し、
  latent の情報保持先が入力割り当てに追従して移るかを確認する。

## 1.2 仮説
- **H12-1 (Role Swap):** `exp1_role_split` で common latent が強く持つ情報（例: category/store）が、`exp2_swapped_split` では specific latent へ移る。
- **H12-2 (Reverse Swap):** `exp1_role_split` で specific latent が強く持つ短期状態情報（例: 直近変動、discount/activity）は、`exp2_swapped_split` で common latent へ移る。
- **H12-3 (Naming Invariance):** branch 名固定ではなく、入力特徴量割り当てが優位に latent 役割を規定する。

---

## 2. 実験対象と比較軸

## 2.1 対象実験
- Scenario9 実装済みの以下 2 条件を使用する。
  - `exp1_role_split`
  - `exp2_swapped_split`

> `exp0_same_input` は補助参照（差の基準）として保持可能だが、Scenario12 の主比較は exp1 vs exp2 とする。

## 2.2 比較の単位
- 実験条件 × latent 種別（common / specific）× probe タスク。
- 主要比較は次の 2 つ。
  1. 同一実験内比較: `score(common_latent)` vs `score(specific_latent)`
  2. 実験間比較: `exp1` と `exp2` で優位 latent が入れ替わるか

---

## 3. Probe タスク設計（Scenario9 の probe を実装化）

## 3.1 構造情報系 probe（本来 common 側に乗りやすい）
- `store_id` 多クラス分類
- `first_category_id` 多クラス分類
- `product_id` 多クラス分類（クラス数が多い場合は top-K に制限）
- `holiday_flag` 2 値分類
- `dow`（曜日）多クラス分類

**期待される入れ替わり:**
- exp1: common > specific
- exp2: specific > common

## 3.2 短期状態系 probe（本来 specific 側に乗りやすい）
- 次時点の売上増減方向（`sale_amount[t+1] - sale_amount[t] > 0`）2 値分類
- 直近変動量ビン分類（`|sale_amount[t]-sale_amount[t-1]|` の quantile bin）
- `discount` 有無 2 値分類
- `activity_flag` 有無 2 値分類

**期待される入れ替わり:**
- exp1: specific > common
- exp2: common > specific

## 3.3 Probe モデル（固定）
- 軽量線形 probe を第一選択:
  - 分類: LogisticRegression（L2、`max_iter=1000`）
  - 回帰を使う場合: Ridge
- 非線形 probe は補助（MLP 1 層）として任意実施。
- 目的は「latent 線形可読性」の比較なので、過度な高容量モデルは避ける。

## 3.4 学習/評価プロトコル
- latent 抽出元モデル本体は凍結（freeze）。
- probe は train split で学習、valid で閾値/設定確認、test で最終評価。
- seed 固定で `n=5`（例: 42/43/44/45/46）を推奨。

---

## 4. 成功判定基準（主張可能ライン）

## 4.1 最低条件
- 少なくとも 2 つ以上の構造情報系 probe で、exp1 と exp2 の優位 latent が逆転する。
- 少なくとも 2 つ以上の短期状態系 probe でも同様に逆転する。

## 4.2 強い条件（推奨）
- 構造系・短期系の双方で、逆転が **平均スコア差 + 統計検定** で確認できる。
- 逆転方向が一貫（例: 構造系は exp1:common 優位 / exp2:specific 優位）。

## 4.3 統計判定（実装レベル）
- seed ごとの差分 `Δ = score(latent_A) - score(latent_B)` を算出。
- exp1 と exp2 の `Δ` の符号反転を確認。
- 95% CI（bootstrap）または対応あり検定（符号付き順位検定）を併記。

---

## 5. 実装計画

## 5.1 追加スクリプト
- 追加先: `scenarios/scenario12_input_assignment_role_swap_probe/run.py`
- 役割:
  1. Scenario9 の学習済みモデル（exp1/exp2）を読み込む
  2. 各 split の `z_common`, `z_specific` を抽出して保存
  3. 各 probe を latent 別に学習・評価
  4. 逆転指標を集計し CSV/図を出力

## 5.2 依存ロジックの再利用
- `src/scenario9_pipeline.py` の split 構築・特徴割り当てルールを再利用。
- Scenario9 の実験名をそのまま利用（`exp1_role_split`, `exp2_swapped_split`）。

## 5.3 CLI 設計（例）
```bash
uv run python scenarios/scenario12_input_assignment_role_swap_probe/run.py \
  --window-size 14 \
  --steps 120 \
  --latent-dim 16 \
  --seeds 42 43 44 45 46 \
  --probes structural temporal \
  --use-scenario9-checkpoints
```

推奨オプション:
- `--reuse-trained`（Scenario9 学習済みを使う）
- `--train-if-missing`（なければ学習）
- `--max-classes-product 100`（product_id 上限）

---

## 6. 出力成果物（固定）

出力先: `scenarios/scenario12_input_assignment_role_swap_probe/output/`

- `probe_scores_long.csv`
  - 列: `experiment, latent, probe_name, split, seed, metric, value`
- `probe_summary.csv`
  - 列: `probe_name, metric, exp1_common, exp1_specific, exp2_common, exp2_specific, swapped(bool)`
- `swap_index.csv`
  - probe ごとの逆転強度（後述）
- `fig_probe_heatmap.png`
  - 実験×latent×probe のスコアヒートマップ
- `fig_swap_direction.png`
  - probe ごとの優位 latent 方向（exp1/exp2）比較
- `scenario12_report.md`
  - 主結果の自動要約（主張可否含む）

---

## 7. 集計指標定義

## 7.1 Probe スコア
- 分類: Accuracy / Macro-F1（クラス不均衡を考慮）
- 回帰: MAE または R2（必要時のみ）

## 7.2 Swap Index（提案）
各 probe について、

- `gap_exp1 = score(exp1, common) - score(exp1, specific)`
- `gap_exp2 = score(exp2, common) - score(exp2, specific)`
- `swap_index = gap_exp1 * gap_exp2`

解釈:
- `swap_index < 0` なら優位方向が逆転（望ましい）
- `|gap|` が大きいほど逆転の強さが高い

---

## 8. 実行手順（オペレーション）

1. Scenario9 の exp1/exp2 チェックポイント有無を確認。
2. 不足があれば Scenario9 を再実行して生成。
3. Scenario12 スクリプトで latent 抽出 + probe 学習を実行。
4. seed 集計と swap 判定を実施。
5. 図表と `scenario12_report.md` を生成。
6. 「言えること / 言えないこと」を更新して記録。

---

## 9. 失敗時の診断フロー

- 逆転が出ない場合:
  - 特徴割り当てが実際に入れ替わっているか（feature list dump）を確認。
  - latent 次元不足/過多を sweep（例: 8/16/32）。
  - probe 学習不足を確認（学習曲線、正則化強度）。
- 片側のみ逆転する場合:
  - 構造系と状態系で難易度差が過大でないか確認。
  - product/store のクラス分布偏りを再点検。

---

## 10. 想定される主張テンプレート

強い結果が得られた場合、以下を主張可能:

- 「common / specific の役割差は branch 名に固定された性質ではない。」
- 「どの特徴量をどちらに入力したかに応じて、latent が保持する情報は系統的に移る。」
- 「したがって、分離は“2本枝を作ったから自然に生じた”のではなく、設計した役割に沿って誘導された。」

---

## 一文まとめ
**Scenario12 は、exp1_role_split と exp2_swapped_split に同一 probe 群を適用し、latent の情報保持先が branch 名ではなく入力特徴量割り当てに追従して入れ替わることを、実装可能な手順と判定基準で示す検証計画である。**
