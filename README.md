# 仮想通貨AI予測システム (最小構成)

## 🚀 すぐに使える機能

### 1. 🏆 取引所スタイルチャート (新機能・推奨)
```bash
streamlit run dashboard/trading_chart_dashboard.py
```
**特徴**: ローソク足、テクニカル指標、ボリューム分析、RSI、MACD

### 2. ⚡ リアルタイムダッシュボード
```bash
streamlit run dashboard/realtime_crypto_dashboard.py
```
**特徴**: 30秒自動更新、ライブ価格、リアルタイム確率表示

### 3. 🔗 実API統合ダッシュボード
```bash
streamlit run dashboard/real_api_trading_dashboard.py
```
**特徴**: Hyperliquid API連携、実データ取得、接続テスト機能

### 4. 📊 固定ダッシュボード
```bash
streamlit run dashboard/crypto_dashboard_fixed.py
```

### 5. 🎯 現在市場予測
```bash
python core/current_market_prediction.py
```

### 6. 🧪 システム性能テスト
```bash
python core/improved_prediction_system.py
```

## 📊 実績

- **平均予測精度**: 50.28%
- **目標達成期間**: 4/7期間 (57.1%)
- **対応銘柄**: BTC, ETH, SOL, AVAX, NEAR, ARB, OP, MATIC

## 📈 取引所スタイルチャート機能

### チャート機能
- **ローソク足チャート**: OHLC データ完全対応
- **テクニカル指標**: MA7/25/99、ボリンジャーバンド
- **ボリューム分析**: 出来高チャート + 移動平均
- **RSI**: 買われすぎ/売られすぎ判定
- **MACD**: トレンド分析 + ヒストグラム

### 時間足対応
- 1分、5分、15分、1時間、4時間、1日

### AI予測統合
- 上昇確率、予測リターン、信頼度
- BUY/SELL/HOLDシグナル

## 🔧 API接続について

### エラー対処
APIエラーが発生する場合:

1. **aiohttp インストール**:
```bash
pip install aiohttp
```

2. **接続テスト**:
```bash
# 実API統合ダッシュボードの「API接続テスト」ボタンを使用
```

3. **フォールバック機能**:
- API接続失敗時は自動的にシミュレートデータに切り替え
- エラーハンドリング強化済み

## 📁 ファイル構造

```
├── core/                           # コアシステム
│   ├── improved_prediction_system.py
│   └── current_market_prediction.py
├── dashboard/                      # WebUIダッシュボード
│   ├── trading_chart_dashboard.py         # 取引所スタイル
│   ├── realtime_crypto_dashboard.py       # リアルタイム更新
│   ├── real_api_trading_dashboard.py      # API統合版
│   └── crypto_dashboard_fixed.py          # 固定版
├── data/historical/                # 履歴データ
├── results/improved_system/        # 性能結果
├── src/api/                        # API クライアント
└── tools/                          # ユーティリティ
```

## 🌐 アクセスURL

各ダッシュボードは異なるポートで起動:

- **取引所チャート**: http://localhost:8504
- **リアルタイム**: http://localhost:8503  
- **API統合**: http://localhost:8505
- **固定版**: http://localhost:8502

## ⚠️ 注意

このシステムは教育・研究目的です。実際の投資には使用しないでください。