# Scenario 3: Latent Demand Recovery with Stockout Mask

## 位置づけ（既存シナリオ拡張か、Scenario 7 として独立か）
本テーマは **Scenario 3 の派生実験** ですが、研究管理上は **Scenario 7（診断シナリオ）として独立** させることを推奨します。

理由:
- Scenario 3 の主目的は「最小の recovery 学習ループが成立するか」の確認。
- 今回の主目的は「なぜ崩れるかを区間別に診断すること」であり、目的が異なる。
- 独立させると、失敗を「モデル失敗」ではなく「censoring 難易度の定量知見」として整理しやすい。

> 実装は Scenario 3 を再利用し、ドキュメントと出力管理だけ Scenario 7 相当に分離する運用が最小コストです。

---

## このシナリオの問い
**「stockout 区間の扱いを変えることで、需要復元タスクの学習ループを作れるか？」** を確認します。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を先に参照してください。

## 1サンプルの具体化（Scenario 3）
- 入力 `x_i`:
  - `hours_sale`
  - `hours_stock_status`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `precpt`
  - `avg_temperature`
  - `avg_humidity`
  - `avg_wind_level`
- マスク `m_i`: `hours_stock_status > 0`
- 目的変数: `hours_sale` を含む入力再構成（マスクで重み調整）

## モデル
- `DecouplingAutoEncoder`
- 損失: 重み付き MSE
  - 非マスク: `1.0 * loss`
  - マスク: `0.1 * loss`
- 評価: WAPE, WPE（`hours_sale` 再構成）

## このシナリオで言えること / 言えないこと
### 言えること
- stockout 情報を利用した recovery 学習の最小実装が可能。
- Scenario 4 の Stage 1 設計の前提を作れる。

### 言えないこと
- 潜在需要の「真値」を復元できたとは断定できない。
- mask重み（0.1）が最適である根拠は未検証。

---

# Experiment 3: Latent Demand Recovery の切り分け（Scenario 7 推奨）

## 目的
`Scenario 3 latent demand recovery` が弱い理由を、
**どの区間で崩れているのか** まで含めて説明可能にする。

現状では Scenario 3 は raw forecast より難しく、
WAPE も大きく、過大予測寄りになっている。
これは単なる失敗ではなく、
**小売の潜在需要復元そのものの難しさ（censoring 問題）** を示す知見候補である。

## 問い
- 誤差はどの区間で大きいのか？
- 欠品近傍が主因なのか？
- マスク重みの置き方が悪いのか？
- 教師信号（proxy）の定義に無理があるのか？

## 条件A: subset evaluation（区間切り分け）
時点 `t` の欠品指標を `s_t \in \{0,1\}`（`1=stockout`）とする。
近傍幅を `\delta`（推奨: `\delta=3` 時間）として、以下の3区間で評価する。

1. **非欠品区間**
   \[
   \mathcal{I}_{\mathrm{non}} = \{t \mid s_t = 0\}
   \]
2. **欠品近傍区間**（前後 `\delta` を含む）
   \[
   \mathcal{I}_{\mathrm{near}} = \{t \mid \exists \tau,\ s_{\tau}=1\ \land\ |t-\tau|\le \delta\}
   \]
3. **全体**
   \[
   \mathcal{I}_{\mathrm{all}} = \{1,\dots,T\}
   \]

> 実装上は `near` が `non` と重なる可能性があるため、
> 必要なら `non_far = non \setminus near` も補助的に出すと解釈が安定する。

## 条件B: mask weight sweep
損失中のマスク重み `\alpha` を sweep:
- `0.1`
- `0.3`
- `0.5`
- `1.0`

重み付き再構成損失（`hours_sale` のみ書くと）:
\[
\mathcal{L}_{\mathrm{rec}}(\alpha)
= \frac{1}{T}\sum_{t=1}^{T}
\left[(1-s_t) + \alpha s_t\right](\hat y_t - y_t)^2
\]

- `\alpha < 1`: 欠品区間を軽く扱う（ノイズ抑制寄り）
- `\alpha = 1`: 欠品/非欠品を同等扱い

## 入力
- current Scenario 3 入力
- stockout indicator `s_t`

## 出力
- latent demand の proxy / recovery 出力 `\hat y_t`

## 評価指標（subset ごと）
評価集合を `\mathcal{I}` として:

1. **WAPE**
\[
\mathrm{WAPE}(\mathcal{I})=
\frac{\sum_{t\in\mathcal{I}}|y_t-\hat y_t|}
{\sum_{t\in\mathcal{I}}|y_t|+\varepsilon}
\]

2. **WPE（符号付きバイアス）**
\[
\mathrm{WPE}(\mathcal{I})=
\frac{\sum_{t\in\mathcal{I}}(\hat y_t-y_t)}
{\sum_{t\in\mathcal{I}}|y_t|+\varepsilon}
\]

3. **train/valid gap**（過学習・不安定性確認）
\[
\Delta_{\mathrm{gap}}(\mathcal{I})=
\mathrm{WAPE}_{\mathrm{valid}}(\mathcal{I})-
\mathrm{WAPE}_{\mathrm{train}}(\mathcal{I})
\]

> `\varepsilon` はゼロ割防止の小定数（推奨 `1e-8`）。

## 追加パラメータ設定（推奨）
Scenario 3 と比較可能性を保つため、以下は固定:
- split: train/valid/test（既存と同一）
- window: `W=14`（主設定）
- optimizer: Adam (`lr=1e-3`)
- epochs または steps: Scenario 3 と同一
- seeds: 最低 `3`（例: `0,1,2`）

可変なのは原則:
- `\alpha \in \{0.1,0.3,0.5,1.0\}`
- 近傍幅 `\delta`（主解析は `3`、感度確認で `1,6` を任意追加）

## 実験表（最小セット）
- 4（mask重み） × 3（subset） × 3（seed）
- レポートは平均 ± 標準偏差

## 成功条件
- 「どこで崩れているか」を subset 別に説明できる
- mask weight に対して単調または解釈可能な変化が出る

## 想定される解釈
### 欠品近傍だけ極端に悪い場合
- 主因は censoring
- latent demand の定義（教師 proxy）の難しさが支配的

### 非欠品区間でも悪い場合
- recovery head そのものが弱い
- タスク定義と表現が噛み合っていない

### mask weight を上げると改善する場合
- 欠品区間の学習寄与が不足していた
- 欠品区間をより重く扱う方針が妥当

### mask weight を上げると悪化する場合
- 欠品 proxy ノイズを過度に学習
- 重み増加で分散が増え、汎化が崩れている

## 実行（運用案）
- 実装は Scenario 3 の学習コードを再利用し、
  `mask_weight` と subset 評価を追加する。
- 出力は `scenario7_latent_demand_recovery_diagnostics` 名で分離保存すると追跡しやすい。

例（CLI 引数を追加した場合）:
```bash
uv run python scenarios/scenario3_latent_demand_recovery/run.py --mask-weight 0.1 --near-window 3
uv run python scenarios/scenario3_latent_demand_recovery/run.py --mask-weight 0.3 --near-window 3
uv run python scenarios/scenario3_latent_demand_recovery/run.py --mask-weight 0.5 --near-window 3
uv run python scenarios/scenario3_latent_demand_recovery/run.py --mask-weight 1.0 --near-window 3
```

## 実行（既存）
```bash
uv run python scenarios/scenario3_latent_demand_recovery/run.py
```
