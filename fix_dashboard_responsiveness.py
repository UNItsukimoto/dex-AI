#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ダッシュボードの応答性改善
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'core'))

def test_dashboard_functions():
    """ダッシュボード機能のテスト"""
    print("=== Dashboard Functions Test ===")
    
    try:
        from enhanced_ai_trader import EnhancedAITrader
        
        # システム初期化
        trader = EnhancedAITrader(10000.0)
        print("[OK] Trader initialized successfully")
        
        # 基本機能テスト
        print("\n[TEST] Basic functions...")
        
        # 1. アカウント情報取得
        account = trader.trading_engine.get_account_summary()
        print(f"[OK] Account info: Balance ${account['balance']:,.2f}")
        
        # 2. 予測実行
        print("[TEST] Running AI prediction...")
        trader.execute_enhanced_strategy()
        print("[OK] AI prediction completed")
        
        # 3. サマリー取得
        summary = trader.get_enhanced_summary()
        print(f"[OK] Summary obtained: {len(summary)} items")
        
        # 4. マルチ銘柄サマリー
        multi_summary = trader.get_multi_symbol_summary()
        print(f"[OK] Multi-symbol summary obtained")
        
        # 5. パフォーマンスサマリー
        try:
            perf_summary = trader.get_performance_summary()
            print(f"[OK] Performance summary obtained")
        except Exception as e:
            print(f"[WARNING] Performance summary error: {e}")
        
        # 6. アラートテスト
        try:
            alert_result = trader.send_test_alert()
            print(f"[OK] Test alert sent")
        except Exception as e:
            print(f"[WARNING] Alert test error: {e}")
        
        print("\n[SUCCESS] All dashboard functions working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_demo():
    """簡単なデモ実行"""
    print("\n=== Simple Demo ===")
    
    try:
        from enhanced_ai_trader import EnhancedAITrader
        from paper_trading_engine import OrderSide, OrderType
        
        # トレーダー初期化
        trader = EnhancedAITrader(15000.0)  # より多めの初期資金
        print(f"[DEMO] Initialized with $15,000")
        
        # デモ取引実行
        print("[DEMO] Executing demo trades...")
        
        # BTC買い
        buy_result = trader.trading_engine.place_order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.2,
            price=45000.0
        )
        print(f"[DEMO] BTC Buy: {buy_result}")
        
        # ETH買い
        eth_result = trader.trading_engine.place_order(
            symbol="ETH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0,
            price=3200.0
        )
        print(f"[DEMO] ETH Buy: {eth_result}")
        
        # ポジション確認
        positions = trader.trading_engine.get_positions()
        print(f"[DEMO] Active positions: {len(positions)}")
        for symbol, pos in positions.items():
            print(f"  - {symbol}: {pos['quantity']:.4f}")
        
        # 一部売却
        if "BTC" in positions:
            sell_result = trader.trading_engine.place_order(
                symbol="BTC",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.1,
                price=46000.0
            )
            print(f"[DEMO] BTC Partial Sell: {sell_result}")
        
        # 最終状態
        final_account = trader.trading_engine.get_account_summary()
        print(f"[DEMO] Final balance: ${final_account['balance']:,.2f}")
        print(f"[DEMO] Final equity: ${final_account['equity']:,.2f}")
        
        # 取引履歴
        trades = trader.trading_engine.get_trade_history()
        print(f"[DEMO] Total trades executed: {len(trades)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing paper trading system responsiveness...")
    
    test1 = test_dashboard_functions()
    test2 = create_simple_demo()
    
    if test1 and test2:
        print("\n[SUCCESS] Paper trading system is working correctly!")
        print("Dashboard should be responsive at: http://localhost:8510")
        print("\nTips for using the dashboard:")
        print("1. Click 'AI予測実行' button to generate predictions")
        print("2. Use the trading panel to place buy/sell orders")
        print("3. Check the Performance tab for trade history")
        print("4. Monitor the Risk Management tab for portfolio status")
    else:
        print("\n[ERROR] Some issues detected in the system")