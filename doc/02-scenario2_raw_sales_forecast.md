# Scenario 2: Raw Sales Forecast (Experiment 1: Forecast Baseline Block)

## 目的
現在の `Scenario 2 raw sales forecast` が、
**本当に意味のある性能なのか** を判断できるようにする。

現状では raw sales forecast は成立しているが、
その値が naive baseline より良いのか、また Decoupling 風表現が役立っているのかがまだ見えにくい。

> 共通定義（1サンプル・global/local・指標）は `doc/00-experiment_problem_setting.md` を参照してください。

## 問い
- `Scenario 2` は naive forecast より良いのか？
- 窓長 14 にする意味はあるのか？
- two-stage にする前に、そもそも素朴な predictor と比べてどうか？

## 入力
- 直近 `W=14` の window tensor
- 現在使っている特徴量一式

## 出力
- 次時点の raw sales

## 比較対象
1. **Last Value**
   - 直近時点の売上をそのまま予測
2. **Moving Average**
   - 直近 `k` 点平均
3. **Flatten + Linear**
   - window を平坦化して線形回帰
4. **Flatten + MLP**
   - window を平坦化して 2〜3 層 MLP
5. **Scenario 2**
   - 現在の raw sales forecast
6. **Scenario 4**
   - current two-stage pipeline

## 評価指標
- valid/test `WAPE`
- valid/test `WPE`

## 成功条件
- `Scenario 2` が Last Value / Moving Average / Linear に対して優位
- 少なくとも naive と同等以下でないこと
- `Scenario 4` が `Scenario 2` に勝てなくても、その差が見えること

## 想定される解釈
### 良い場合
- Decoupling 風構成は naive baseline を超える
- 小売時系列にも一定の表現学習の価値がある

### 悪い場合
- 直近値や単純平均で足りている
- まだ Decoupling 的な構成の優位性は弱い
- 特徴量またはタスク設計の見直しが必要

## 実行
```bash
uv run python scenarios/scenario2_raw_sales_forecast/run.py
```
