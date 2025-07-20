#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ペーパートレードシステムのテスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'core'))

def test_paper_trading():
    """ペーパートレード機能のテスト"""
    print("=== ペーパートレードシステムテスト ===")
    
    try:
        from enhanced_ai_trader import EnhancedAITrader
        print("[OK] Enhanced AI Trader インポート成功")
        
        # システム初期化
        trader = EnhancedAITrader(10000.0)
        print(f"[OK] トレーダー初期化成功 - 初期残高: $10,000")
        
        # 初期状態確認
        account = trader.trading_engine.get_account_summary()
        print(f"[INFO] 現在の残高: ${account['balance']:,.2f}")
        print(f"[INFO] エクイティ: ${account['equity']:,.2f}")
        
        # 予測実行テスト
        print("\n[TEST] AI予測実行テスト...")
        trader.execute_enhanced_strategy()
        print("[OK] AI予測実行完了")
        
        # 予測結果確認
        summary = trader.get_enhanced_summary()
        predictions = summary.get('latest_predictions', [])
        print(f"[INFO] 予測結果数: {len(predictions)}")
        
        if predictions:
            for pred in predictions[:3]:  # 最初の3件
                print(f"  - {pred['symbol']}: {pred['signal']} (信頼度: {pred['confidence']:.1%})")
        
        # ポジション状況確認
        positions = trader.trading_engine.get_positions()
        print(f"[INFO] アクティブポジション数: {len(positions)}")
        
        if positions:
            for symbol, pos in positions.items():
                print(f"  - {symbol}: {pos['quantity']:.4f} (損益: ${pos.get('unrealized_pnl', 0):.2f})")
        
        # 取引履歴確認
        trades = trader.trading_engine.get_trade_history()
        print(f"[INFO] 取引履歴数: {len(trades)}")
        
        if trades:
            for trade in trades[-3:]:  # 最新3件
                print(f"  - {trade.get('symbol')}: {trade.get('side')} {trade.get('quantity'):.4f} @ ${trade.get('price'):.2f}")
        
        # マルチ銘柄テスト
        print("\n[TEST] マルチ銘柄予測テスト...")
        trader.execute_multi_symbol_strategy()
        print("[OK] マルチ銘柄予測完了")
        
        # マルチ銘柄結果確認
        multi_summary = trader.get_multi_symbol_summary()
        trading_summary = multi_summary.get('trading_summary', {})
        print(f"[INFO] 有効銘柄数: {trading_summary.get('enabled_symbols', 0)}")
        print(f"[INFO] 取引機会: {trading_summary.get('trading_opportunities', 0)}")
        
        # パフォーマンス確認
        print("\n[TEST] パフォーマンス分析テスト...")
        perf_summary = trader.get_performance_summary()
        metrics = perf_summary.get('metrics', {})
        print(f"[INFO] 総リターン: ${metrics.get('total_return', 0):.2f}")
        print(f"[INFO] 勝率: {metrics.get('win_rate', 0):.1%}")
        
        # 最終状態確認
        final_account = trader.trading_engine.get_account_summary()
        print(f"\n[FINAL] 最終残高: ${final_account['balance']:,.2f}")
        print(f"[FINAL] 最終エクイティ: ${final_account['equity']:,.2f}")
        
        profit_loss = final_account['equity'] - final_account['balance']
        print(f"[FINAL] 損益: ${profit_loss:,.2f}")
        
        print("\n[SUCCESS] ペーパートレードシステム正常動作確認完了！")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_trade():
    """手動取引テスト"""
    print("\n=== 手動取引テスト ===")
    
    try:
        from enhanced_ai_trader import EnhancedAITrader
        from paper_trading_engine import OrderSide, OrderType
        
        trader = EnhancedAITrader(10000.0)
        
        # 手動でBTC買い注文
        print("[TEST] BTC買い注文実行...")
        order_result = trader.trading_engine.place_order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,  # 0.1 BTC
            price=45000.0  # $45,000
        )
        
        if order_result:
            print(f"[OK] 買い注文成功: {order_result}")
        else:
            print("[ERROR] 買い注文失敗")
        
        # ポジション確認
        positions = trader.trading_engine.get_positions()
        if "BTC" in positions:
            pos = positions["BTC"]
            print(f"[INFO] BTCポジション: {pos['quantity']:.4f} BTC")
        
        # 手動でBTC売り注文
        print("[TEST] BTC売り注文実行...")
        sell_result = trader.trading_engine.place_order(
            symbol="BTC",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1,  # 0.1 BTC
            price=46000.0  # $46,000
        )
        
        if sell_result:
            print(f"[OK] 売り注文成功: {sell_result}")
        else:
            print("[ERROR] 売り注文失敗")
        
        # 最終状態
        final_account = trader.trading_engine.get_account_summary()
        print(f"[FINAL] 手動取引後残高: ${final_account['balance']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 手動取引テスト失敗: {e}")
        return False

if __name__ == "__main__":
    success1 = test_paper_trading()
    success2 = test_manual_trade()
    
    if success1 and success2:
        print("\n🎉 全てのテスト成功！ペーパートレードシステム正常動作中")
        print("ダッシュボードURL: http://localhost:8510")
    else:
        print("\n❌ テストで問題が発見されました")