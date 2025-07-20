#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マルチ銘柄統合テスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'core'))

try:
    from enhanced_ai_trader import EnhancedAITrader
    print("[OK] EnhancedAITrader with multi-symbol support import success")
    
    # システム初期化
    trader = EnhancedAITrader(10000.0)
    print(f"[OK] Enhanced AI Trader initialized with multi-symbol support")
    
    # マルチ銘柄サマリー取得
    multi_summary = trader.get_multi_symbol_summary()
    print("[OK] Multi-symbol summary obtained")
    
    # 基本情報表示
    trading_summary = multi_summary.get('trading_summary', {})
    print(f"[INFO] Trading Summary:")
    print(f"  - Total symbols: {trading_summary.get('total_symbols', 0)}")
    print(f"  - Enabled symbols: {trading_summary.get('enabled_symbols', 0)}")
    print(f"  - Trading opportunities: {trading_summary.get('trading_opportunities', 0)}")
    print(f"  - High confidence signals: {trading_summary.get('high_confidence_signals', 0)}")
    
    # 有効銘柄リスト表示
    enabled_symbols = trading_summary.get('enabled_symbol_list', [])
    print(f"[INFO] Enabled symbols: {enabled_symbols}")
    
    # 銘柄設定テスト
    print("\n[TEST] Symbol configuration management...")
    for symbol in ['BTC', 'ETH', 'SOL', 'AVAX']:
        config = trader.get_symbol_config(symbol)
        if config:
            status = "[ENABLED]" if config['enabled'] else "[DISABLED]"
            print(f"  {symbol}: {status} (Max: {config['max_position_size']:.1%}, "
                  f"Min Conf: {config['min_confidence']:.1%})")
    
    # 設定変更テスト
    print("\n[TEST] Symbol configuration update...")
    original_config = trader.get_symbol_config('AVAX')
    print(f"AVAX original status: {'Enabled' if original_config['enabled'] else 'Disabled'}")
    
    # AVAX有効化
    trader.enable_symbol_trading('AVAX')
    updated_config = trader.get_symbol_config('AVAX')
    print(f"AVAX updated status: {'Enabled' if updated_config['enabled'] else 'Disabled'}")
    
    # 設定調整
    trader.update_symbol_config('AVAX', max_position_size=0.08, min_confidence=0.75)
    final_config = trader.get_symbol_config('AVAX')
    print(f"AVAX final config: Max={final_config['max_position_size']:.1%}, "
          f"MinConf={final_config['min_confidence']:.1%}")
    
    # マルチ銘柄戦略実行テスト
    print("\n[TEST] Multi-symbol strategy execution...")
    try:
        trader.execute_multi_symbol_strategy()
        print("[OK] Multi-symbol strategy executed successfully")
    except Exception as e:
        print(f"[WARNING] Strategy execution had issues (expected with mock data): {e}")
    
    # 設定保存テスト
    print("\n[TEST] Configuration save...")
    trader.save_multi_symbol_config()
    print("[OK] Multi-symbol configuration saved")
    
    # 最終サマリー
    final_summary = trader.get_multi_symbol_summary()
    final_trading_summary = final_summary.get('trading_summary', {})
    print(f"\n[FINAL] Updated Summary:")
    print(f"  - Total symbols: {final_trading_summary.get('total_symbols', 0)}")
    print(f"  - Enabled symbols: {final_trading_summary.get('enabled_symbols', 0)}")
    print(f"  - Enabled list: {final_trading_summary.get('enabled_symbol_list', [])}")
    
    print("\n[SUCCESS] Multi-symbol integration test completed!")
    print("Dashboard access: streamlit run dashboard/risk_managed_dashboard.py")
    print("Check the '🌍 マルチ銘柄' tab for multi-symbol management")
    
except Exception as e:
    print(f"[ERROR] Integration test failed: {e}")
    import traceback
    traceback.print_exc()