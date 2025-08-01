# 📈 ペーパートレーディングテストガイド

## 🎯 テストの目的
- AIの予測精度を確認
- リスク管理システムの動作確認
- 取引戦略の有効性を検証
- UIの使いやすさを評価

## 🚀 Step 1: プラットフォームの起動

### コマンドライン操作
```bash
cd "C:\Projects\hyperliquid-prediction-ai\dex-AI"
streamlit run dashboard/unified_trading_platform.py --server.port=8506
```

### ブラウザアクセス
- URL: http://localhost:8506
- 推奨ブラウザ: Chrome, Edge, Firefox

## ⚙️ Step 2: 初期設定

### サイドバー設定
1. **🎯 取引モード**: "🚀 リアル体験モード"
2. **📈 メイン銘柄**: BTC, ETH, SOL から選択
3. **⏰ チャート時間枠**: 1h または 4h（推奨）
4. **🔄 自動更新**: 有効（10秒間隔）

### アカウント確認
- 初期残高: $10,000
- エクイティ: $10,000
- 日次損益: $0

## 📊 Step 3: 基本操作の確認

### 3.1 チャート分析
- [ ] 価格チャートが正常に表示される
- [ ] ローソク足データが更新される
- [ ] 移動平均線が表示される
- [ ] ボリュームが表示される

### 3.2 AI予測の確認
- [ ] 予測シグナル（BUY/SELL/HOLD）が表示される
- [ ] 信頼度が表示される（0-100%）
- [ ] 予測確率が表示される
- [ ] 予測根拠が確認できる

### 3.3 リスク管理の確認
- [ ] リスクレベルが表示される
- [ ] 総エクスポージャーが計算される
- [ ] ポートフォリオボラティリティが表示される
- [ ] 推奨ポジションサイズが計算される

## 💰 Step 4: 実際の取引テスト

### 4.1 買い注文テスト
1. **取引**タブを選択
2. **注文タイプ**: 成行注文
3. **注文方向**: 買い
4. **金額**: $100-500（小額テスト）
5. **注文実行**ボタンをクリック

**確認項目:**
- [ ] 注文が正常に実行される
- [ ] 残高が正しく減る
- [ ] ポジションが作成される
- [ ] 取引履歴に記録される

### 4.2 売り注文テスト
1. 保有ポジションがある状態で
2. **注文方向**: 売り
3. **数量**: 保有数量の一部または全部
4. **注文実行**

**確認項目:**
- [ ] 売り注文が実行される
- [ ] 損益が計算される
- [ ] 残高が更新される
- [ ] ポジションが減る/クローズされる

### 4.3 複数銘柄テスト
1. BTC, ETH, SOL で各1回ずつ取引
2. 各銘柄の特性を確認
3. ポートフォリオの分散効果を確認

## 📈 Step 5: 高度機能のテスト

### 5.1 AI予測に基づく取引
1. AIがBUYシグナルを出した時に買い注文
2. 信頼度が70%以上の予測を優先
3. 利益確定・損切りの自動化を確認

### 5.2 リスク管理機能
1. **ポジションサイズ**: 推奨サイズでの取引
2. **ストップロス**: 自動設定の確認
3. **テイクプロフィット**: 自動設定の確認
4. **リスク制限**: 制限に達した時の動作確認

### 5.3 パフォーマンス分析
1. **取引履歴**: 全取引の記録確認
2. **損益グラフ**: 時系列での損益推移
3. **勝率**: 勝ちトレードの割合
4. **最大ドローダウン**: 最大損失幅

## ✅ Step 6: テスト結果の評価

### 6.1 技術面の評価
- [ ] システムの安定性（エラーなし）
- [ ] データの正確性（価格、残高等）
- [ ] 応答速度（操作の遅延なし）
- [ ] UI/UXの使いやすさ

### 6.2 AI予測の評価
- [ ] 予測精度（正解率）
- [ ] 信頼度の妥当性
- [ ] シグナルのタイミング
- [ ] 市場状況への適応性

### 6.3 リスク管理の評価
- [ ] ポジションサイズの適切性
- [ ] ストップロスの有効性
- [ ] ドローダウンの制御
- [ ] 資金管理の妥当性

## 🎯 Step 7: 実戦シナリオテスト

### シナリオ1: トレンド相場での取引
- 強い上昇/下降トレンド時の動作確認
- AIの順張り戦略の有効性

### シナリオ2: レンジ相場での取引
- 横ばい相場でのAI判断
- 無駄な売買の回避能力

### シナリオ3: 急激な価格変動時
- 突発的な価格急騰/急落への対応
- リスク管理の発動確認

## 📊 テスト期間の推奨

### 短期テスト（1-3日）
- 基本機能の動作確認
- UIの使いやすさ確認
- 簡単な取引テスト

### 中期テスト（1-2週間）
- AI予測精度の評価
- 様々な市場状況での確認
- パフォーマンス分析

### 長期テスト（1ヶ月以上）
- 戦略の安定性確認
- 長期的な収益性評価
- システムの堅牢性確認

## 🚨 注意事項

### 安全な使用のために
- ペーパートレーディングは仮想資金
- 実際の資金は使用されない
- API接続はテストネットのみ
- 実取引前に十分なテストを実施

### データについて
- 価格データは実際の市場データ
- 取引コストも現実的に設定
- スリッページや遅延も再現

## 📞 問題が発生した場合

### エラー対処
1. ブラウザの再読み込み
2. アプリケーションの再起動
3. エラーログの確認
4. システム設定の見直し

### サポート情報
- エラーメッセージのスクリーンショット
- 操作手順の記録
- システム環境の情報

---

## 🎉 テスト完了後

テストが完了したら、以下を確認してください：

1. **結果の記録**: 取引履歴、損益、勝率等
2. **改善点の洗い出し**: UIやAI予測の改善提案
3. **実取引への移行判断**: ペーパーテストの結果に基づく判断

**成功の目安:**
- 勝率: 60%以上
- 最大ドローダウン: 5%以下
- シャープレシオ: 1.0以上

これらの基準を満たせば、実取引への移行を検討できます。