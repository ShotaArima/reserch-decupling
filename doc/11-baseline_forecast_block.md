# Baseline Forecast Block（共通ベースライン集約）

## 目的
Scenario 2 で使っていた予測比較群を `baselines/forecast_block` に集約し、
「何を baseline と見なしているか」を一箇所で説明できるようにする。

## 対象モデル

### Naive 系
1. **LastValue**
   - 直近時点の `sale_amount` をそのまま次時点予測に使う。
2. **MovingAverage**
   - `k in {3,7,14}` の移動平均を候補とし、valid WAPE 最小の `k` を採用。

### タブラー回帰系
3. **FlattenLinear**
   - window tensor を flatten して線形回帰。
4. **FlattenMLP**
   - flatten 入力に MLP を適用（`[128,64]` / `[128,64,32]` から valid WAPE 最良を選択）。

### 時系列統計モデル
5. **Prophet**
   - 各サンプル window の `sale_amount` を日次系列とみなし、1-step 先を予測。
   - `prophet` 未インストール時はスキップ（他モデルは継続）。

### Decoupling 系
6. **Scenario2**
   - `DecouplingAutoEncoder + ForecastHead`。
7. **Scenario4**
   - Recovery（Stage1）+ Forecast（Stage2）の2段構成。

## 実行
```bash
uv run python baselines/forecast_block/run.py
```

後方互換として以下でも同じ処理が走る。
```bash
uv run python scenarios/scenario2_raw_sales_forecast/run.py
```

## 出力
- `baselines/forecast_block/outputs/forecast_baseline_results.csv`
- 各学習曲線 PNG
- valid/test WAPE 比較棒グラフ
