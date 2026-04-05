# Scenario 6: Local / Global / Both Ablation (Experiment 2)

## 位置づけ（Scenario 2 を補強するか、独立シナリオ化するか）
本実験は **Scenario 2 の結論を強化するための追加実験** ですが、
研究管理上は **Scenario 6 として独立** させることを推奨します。

理由:
- Scenario 2 は「raw sales forecast が naive より有効か」の検証が主目的。
- 本実験は「予測に効いている情報源が local / global のどちらか」の **表現分離検証** が主目的。
- 主目的が異なるため、シナリオを分けた方が結果の解釈と失敗時の切り分けが明確になる。

> ただし実装資産（データ split、学習条件、評価関数）は Scenario 2 を最大限再利用する。

---

## 目的
現在のモデルで得られている latent が、
**本当に役割差を持っていそうか** を最低限確認する。

現時点では再構成はできていても、
global / local の意味的分離までは示せていない。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を参照してください。

## 問い
- forecast に効いているのは local か global か？
- both（local + global）を使う意味はあるか？
- 分離表現として最低限役割差が出ているか？

## 条件（3-way ablation）
1. **local only**
2. **global only**
3. **local + global**

## 入力
- Scenario 2 もしくは Scenario 4 Stage 2 と同様の window 入力
- latent は以下のどちらかで供給
  - **A案（推奨）**: Scenario 2 の end-to-end で得る latent
  - **B案（比較用）**: Scenario 4 Stage 1 で得た latent を固定して Stage 2 のみ学習

### 推奨運用
- まず A案（Scenario 2 latent）で実施し、最小の検証を完了。
- 余力があれば B案（Stage 1 latent）を追加し、
  「end-to-end 表現」と「two-stage 表現」で ablation の傾向が一致するか確認する。

## 出力
- 次時点 raw sales（`x_t -> y_{t+1}`）

## モデル仕様（最小差分で公平比較）
`ForecastHead` は条件間で可能な限り共通化し、
**入力に使う latent のみ切り替える**。

- local only:
  - 入力: `local_i`
  - 実装例: `ForecastHead(local_dim=16, global_dim=0, horizon=1)` 相当
- global only:
  - 入力: `global_i`
  - 実装例: `ForecastHead(local_dim=0, global_dim=16, horizon=1)` 相当
- both:
  - 入力: `[local_i, global_i]`
  - 実装例: 現行仕様（`local_dim=16, global_dim=16`）

### 学習条件（固定）
- split: Scenario 2 と同一 train/valid/test
- window: `W=14`（主設定）
- 損失: L1Loss
- optimizer: Adam (`lr=1e-3`)
- step: `100`
- seed: 同一 seed を複数回（例: 3 seeds）

## 評価指標
- valid/test `WAPE`
- valid/test `WPE`

## 成功条件
- `local only` と `global only` の性能差が見える
- `local + global` が最良、または少なくとも片方単独より悪くない

## 想定される解釈
### `local only` が強い場合
- 直近状態の方が支配的
- 現段階では dynamic information が主要

### `global only` がそこそこ強い場合
- 店舗×商品の構造差をある程度保持できている
- global が単なる余剰表現ではない

### `both` が最良の場合
- 分離して保持する意味がある
- 「役割別 latent」の主張の足場になる

### 3条件がほぼ同じ場合
- 現状の分離はまだ弱い
- latent の役割設計か正則化が足りない

## Scenario 2 / 4 との関係
- Scenario 2: 「予測精度が出るか」を確認（性能の有無）
- Scenario 4: 「recovery を挟む二段設計の成立性」を確認（パイプライン妥当性）
- Scenario 6: 「local/global の役割差」を確認（表現分離妥当性）

3つを合わせることで、
1) 予測できる、2) パイプライン設計できる、3) 役割差がある、
という主張を段階的に積み上げられる。

## 実行（想定）
Scenario 6 を追加する場合の実行例:

```bash
uv run python scenarios/scenario6_local_global_ablation/run.py
```

> まずは Scenario 2 の `run.py` にフラグ追加でも実行可能。
> ただし再現実験としては `scenario6_*` ディレクトリを独立させた方が管理しやすい。
