# Scenario 19: Prophet / VAE Decoupling の時間軸拡張比較

## 目的
既存の 14 日予測ベースラインを拡張し、**21 / 28 / 35 / 42 日**を追加して、
Prophet と VAE 系 Decoupling（Scenario2）の予測誤差の増え方を比較・可視化する。

- 対象 horizon: `14, 21, 28, 35, 42`
- 対象モデル:
  - Prophet（各 window を単変量時系列として next-step 予測）
  - Scenario2（DecouplingAutoEncoder + ForecastHead）
- 指標:
  - `valid_wape`, `valid_wpe`
  - `test_wape`, `test_wpe`

---

## 実装
実験コードは以下に配置。

- `scenarios/scenario19_prophet_vae_horizon_extension/run.py`

このスクリプトは horizon ごとに以下を実行する。

1. window 作成・split・正規化
2. Prophet 推論
3. Scenario2 学習・推論
4. 指標計算・保存
5. 可視化（horizon x error）

---

## 実行方法
```bash
uv run python scenarios/scenario19_prophet_vae_horizon_extension/run.py
```

### 8GB GPU での推奨実行
Prophet は CPU 計算が中心で時間がかかるため、GPU で高速化したい場合は
Scenario2 側を CUDA で回しつつ、Prophet サンプル数を制限する。

```bash
uv run python -u scenarios/scenario19_prophet_vae_horizon_extension/run.py \
  --device cuda \
  --batch-size 128 \
  --train-steps 80 \
  --prophet-max-samples 128 \
  --log-interval 10 \
  | tee scenarios/scenario19_prophet_vae_horizon_extension/output/experiments.log
```

さらに速度優先で動作確認だけしたい場合は Prophet をスキップできる。

```bash
uv run python -u scenarios/scenario19_prophet_vae_horizon_extension/run.py \
  --device cuda \
  --batch-size 128 \
  --train-steps 80 \
  --skip-prophet \
  | tee scenarios/scenario19_prophet_vae_horizon_extension/output/experiments.log
```

---

## 出力先
`scenarios/scenario19_prophet_vae_horizon_extension/output/` に以下を保存。

- `horizon_extension_results.csv`
- `horizon_extension_valid_wape.png`
- `horizon_extension_test_wape.png`
- `h14_scenario2_train_loss.png`
- `h21_scenario2_train_loss.png`
- `h28_scenario2_train_loss.png`
- `h35_scenario2_train_loss.png`
- `h42_scenario2_train_loss.png`
- `experiments.log`

---

## 実験結果（2026-04-19 〜 2026-04-20 実行）
`scenarios/scenario19_prophet_vae_horizon_extension/output/experiments.log` の結果を整理。

### WAPE（horizon 別）
| Horizon | Model | valid_wape | test_wape |
|---:|---|---:|---:|
| 14 | Prophet | 0.4636 | 0.4584 |
| 14 | Scenario2 | 0.4292 | 0.4242 |
| 21 | Prophet | 0.4636 | 0.4584 |
| 21 | Scenario2 | 0.4281 | 0.4229 |
| 28 | Prophet | 0.4636 | 0.4584 |
| 28 | Scenario2 | 0.4284 | 0.4235 |
| 35 | Prophet | 0.4636 | 0.4584 |
| 35 | Scenario2 | 0.4283 | 0.4231 |
| 42 | Prophet | 0.4636 | 0.4584 |
| 42 | Scenario2 | 0.4285 | 0.4232 |

### 主要な観察
- **全 horizon で Scenario2 が Prophet を上回る**（valid/test ともに低い WAPE）。
- Scenario2 の test_wape は `0.4229〜0.4242` の狭い範囲で安定しており、
  horizon を 14→42 日に伸ばしても大きな劣化は見られない。
- Prophet 側は全 horizon で同一値（`valid_wape=0.4636, test_wape=0.4584`）となっており、
  今回の設定では horizon 拡張に対する差が出ていない。

### 参考: Scenario2 学習 loss（start / mid / end）
| Horizon | start | mid | end |
|---:|---:|---:|---:|
| 14 | 1.0013 | 0.4223 | 0.4173 |
| 21 | 0.9680 | 0.4204 | 0.4160 |
| 28 | 1.0602 | 0.4211 | 0.4165 |
| 35 | 1.0134 | 0.4178 | 0.4160 |
| 42 | 0.8382 | 0.4190 | 0.4160 |

---

## まとめ
- Scenario19 の範囲では、**Decoupling（Scenario2）は horizon 拡張に対して安定**であり、
  Prophet baseline より一貫して良い精度を示した。
- 次アクションとしては、Prophet 側の horizon 感度が出にくい要因（サンプル抽出・前処理・推論設定）を
  切り分けると、比較の解像度がさらに上がる。
