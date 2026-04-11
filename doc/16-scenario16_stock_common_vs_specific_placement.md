# Scenario 16: stock 系特徴の common / specific 配置検証（実装計画）

## 位置づけ
Scenario16 は Scenario10 の知見を、**段階的追加 + 対照実験**で再検証するための設計です。

- Scenario10: stock 系特徴を specific 側に追加して有効性を確認
- Scenario16: 「どちらに置くのが自然か」を、同一条件で明示比較

本シナリオの主眼は、
**「stock 系は short-term state として specific 側に置く方が自然か」**を、
性能・ablation・probe の 3 軸で強く示すことです。

---

## 1. 目的

1. stock 系特徴を段階的に追加したときの性能変化を定量化する。
2. stock 系特徴を common 側に置いた対照条件と比較し、配置妥当性を検証する。
3. probe により、stock 状態情報がどの latent（common/specific）に強く載るかを確認する。

---

## 2. 研究質問（RQ）

- **RQ16-1:** `stock_hour6_22_cnt` は specific 側へ追加したときに改善するか。
- **RQ16-2:** `hours_stock_status` を追加すると、specific 側の改善はさらに強まるか。
- **RQ16-3:** 同じ stock 系を common 側へ入れた場合、specific 側ほど改善するか。
- **RQ16-4:** stock 状態の probe は specific latent で高スコアを示すか。

---

## 3. 実験条件（必須）

Scenario9 Exp-1（stock 系なし）を基準として、以下を比較する。

## Exp-16A: specific へ 1 特徴のみ追加
- 追加先: **specific**
- 追加特徴: `stock_hour6_22_cnt` のみ
- 意図: stock 量的特徴単体の寄与を切り出す

## Exp-16B: specific へ 2 特徴追加
- 追加先: **specific**
- 追加特徴: `stock_hour6_22_cnt`, `hours_stock_status`
- 意図: Scenario10 相当の構成を再確認

## Exp-16C: common へ 2 特徴追加（対照）
- 追加先: **common**
- 追加特徴: `stock_hour6_22_cnt`, `hours_stock_status`
- 意図: 「stock を common に置いても同等か」を反証的に確認

> 比較の中心は **Exp-16B vs Exp-16C**。

---

## 4. 仮説

- **H16-1:** Exp-16A は baseline 比で悪化せず、subset により改善が出る。
- **H16-2:** Exp-16B は Exp-16A 以上に改善（特に stockout subset）を示す。
- **H16-3:** Exp-16C の改善幅は Exp-16B より小さい、または不安定。
- **H16-4:** stock 状態の probe スコアは common latent より specific latent で高い。

---

## 5. 実装スコープ

## 5.1 変更対象（推奨）
- `scenarios/scenario10_stock_extension_local_branch/run.py` をベースに Scenario16 用 run を新設
- 出力先は `scenarios/scenario16_stock_common_vs_specific/outputs/` に固定
- 実験計画書は本ファイル（`doc/16-...md`）を正とする

## 5.2 実装方針
- 特徴量生成ロジックは既存関数を再利用し、**追加先（common/specific）だけを切替可能**にする。
- 条件差は config で管理し、コード分岐の重複を避ける。
- 学習・評価コードは同一で、feature assignment のみを変える。

---

## 6. コンフィグ設計（実装レベル）

`experiment_configs` を以下のように定義する。

```python
experiment_configs = {
    "baseline_s9_exp1": {
        "stock_to_specific": [],
        "stock_to_common": [],
    },
    "exp16a_specific_stock_cnt_only": {
        "stock_to_specific": ["stock_hour6_22_cnt"],
        "stock_to_common": [],
    },
    "exp16b_specific_stock_both": {
        "stock_to_specific": ["stock_hour6_22_cnt", "hours_stock_status"],
        "stock_to_common": [],
    },
    "exp16c_common_stock_both": {
        "stock_to_specific": [],
        "stock_to_common": ["stock_hour6_22_cnt", "hours_stock_status"],
    },
}
```

### 実装ルール
- 同一 seed 群を全条件で共有（例: `seeds=[42, 52, 62]`）。
- window/horizon/optimizer/lr/epoch は全条件固定。
- 学習終了条件（early stopping patience）も固定。

---

## 7. データ処理仕様（固定）

1. split は Scenario9/10 と同一。
2. 正規化は train fit のみ、valid/test へ適用。
3. `hours_stock_status` がカテゴリの場合は符号化方式を固定（one-hot か embedding のいずれか）。
4. 欠損処理は全条件同一（ゼロ埋め + mask など）で統一。
5. 推論時に利用できない特徴は学習でも使用しない。

---

## 8. 評価設計

## 8.1 主指標
- WAPE
- WPE
- MAE

## 8.2 subset 指標
- all
- stockout
- non-stockout

## 8.3 追加評価（必須）
- latent ablation: `both / common-only / specific-only`
- probe（stock 状態関連）:
  - `hours_stock_status` 識別
  - stockout 有無識別
  - 次時点 stock 逼迫の二値判別（定義可能なら）

---

## 9. 出力物仕様

`scenarios/scenario16_stock_common_vs_specific/outputs/` に以下を保存。

- `scenario16_metrics_overall.csv`
- `scenario16_metrics_by_subset.csv`
- `scenario16_ablation.csv`
- `scenario16_probe_scores.csv`
- `scenario16_summary.md`

`scenario16_summary.md` には次を必須記載:
- Baseline / Exp-16A / Exp-16B / Exp-16C の比較表
- Exp-16B vs Exp-16C の差分（絶対値・相対値）
- 仮説 H16-1〜H16-4 の採否

---

## 10. 判定基準（強い結果）

以下を満たすほど「stock は specific が自然」を強く主張できる。

1. **specific-only が改善**
   - Exp-16B の `specific-only` が baseline より改善
2. **common 追加の改善が弱い**
   - Exp-16C の改善幅が Exp-16B より小さい
3. **probe で specific 優位**
   - stock 状態タスクで `z_specific` のスコアが `z_common` を上回る

---

## 11. 実行手順（オペレーション）

1. Baseline（Scenario9 Exp-1 相当）を再実行して基準値を保存
2. Exp-16A 実行
3. Exp-16B 実行
4. Exp-16C 実行
5. 各条件で ablation 実行
6. 各条件で probe 実行
7. subset 集計（all / stockout / non-stockout）
8. `scenario16_summary.md` を生成

---

## 12. リスクと対処

- stock 特徴品質が低い場合
  - 欠損率・外れ値・更新遅延を監査
- seed 感度が高い場合
  - seed 数を 3→5 に増やし、平均 + 分散で判断
- 指標が拮抗する場合
  - subset と probe を優先して配置妥当性を解釈

---

## 13. 一文まとめ

**Scenario16 は、stock 系特徴を specific へ段階追加する条件と common へ置く対照条件を同一設定で比較し、stock 系が short-term state として specific latent に載ることを性能・ablation・probe で検証する実装実験である。**

---

## 実験結果（記録済みログ）

- ログ:
  - [output.log](../scenarios/scenario16_stock_common_vs_specific/output/output.log)
- 出力:
  - [scenario16_summary.md](../scenarios/scenario16_stock_common_vs_specific/output/scenario16_summary.md)
  - [scenario16_metrics_overall.csv](../scenarios/scenario16_stock_common_vs_specific/output/scenario16_metrics_overall.csv)
  - [scenario16_metrics_by_subset.csv](../scenarios/scenario16_stock_common_vs_specific/output/scenario16_metrics_by_subset.csv)
  - [scenario16_probe_scores.csv](../scenarios/scenario16_stock_common_vs_specific/output/scenario16_probe_scores.csv)
- 結果メモ（test WAPE の例）:
  - `baseline_s9_exp1 / both`（seed平均）: おおむね `0.43` 付近
  - `exp16a_specific_stock_cnt_only / specific_only`: seed42=`0.4532`, seed62=`0.4535` まで改善
  - `exp16c_common_stock_both / both` は seed52 で `0.4561` と悪化ケースあり
