# Scenario 1: Representation Probe

## 目的
FreshRetailNet-50K 上で、**global 表現**と**local 表現**の分離が最低限成立しそうかを、再構成ベースで確認する最小実験です。

## 背景と狙い
- 時系列の中には「比較的安定な要因（global）」と「時変要因（local）」が混在します。
- まずは予測精度ではなく、分離表現の学習が破綻していないかを確認します。
- 成功条件は、過学習せずに再構成誤差が下がり、後続タスクへ進める基盤が得られることです。

## 入力
- 対象データセット: `FreshRetailNet/FreshRetailNet-50K`（Hugging Face Datasets）
- 主な特徴量（スクリプトに準拠）
  - `sale_amount`
  - `stock_hour6_22_cnt`
  - `discount`
  - `holiday_flag`
  - `activity_flag`
  - `precpt`
  - `avg_temperature`
  - `avg_humidity`
  - `avg_wind_level`

## モデル構成
- `src.models.DecouplingAutoEncoder` を利用
  - local encoder
  - global encoder
  - decoder
- 損失: MSE 再構成損失
- 最小反復（サンプル実装）で挙動確認

## 出力
- `reconstruction_mse`

## 何を確認するか
- 再構成誤差が妥当な方向に下がるか
- 学習が数ステップでも崩壊せず進むか
- 以降のシナリオ（forecast/recovery/counterfactual）に使える表現学習の土台になるか

## 実行
```bash
uv run python scenarios/scenario1_representation_probe/run.py
```

## 次の発展
- local/global それぞれの潜在に対して、属性分類の probe を追加
- 埋め込み可視化（UMAP/PCA）で分離傾向を定性確認
