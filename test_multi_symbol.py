#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マルチ銘柄システムテスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'core'))

try:
    from multi_symbol_manager import MultiSymbolManager
    print("[OK] MultiSymbolManager import success")
    
    # システム初期化
    manager = MultiSymbolManager()
    print(f"[OK] MultiSymbol Manager initialized")
    
    # 基本情報表示
    all_symbols = manager.get_all_symbols()
    enabled_symbols = manager.get_enabled_symbols()
    
    print(f"[INFO] Total symbols: {len(all_symbols)}")
    print(f"[INFO] Enabled symbols: {len(enabled_symbols)}")
    print(f"[INFO] Enabled list: {enabled_symbols}")
    
    # 設定テンプレート出力
    manager.export_config_template()
    print("[OK] Configuration template exported")
    
    # 取引サマリー生成
    summary = manager.generate_trading_summary()
    print("[OK] Trading summary generated:")
    print(f"  - Total symbols: {summary['total_symbols']}")
    print(f"  - Enabled symbols: {summary['enabled_symbols']}")
    print(f"  - Trading opportunities: {summary['trading_opportunities']}")
    
    # 銘柄設定例
    print("\n[TEST] Symbol configuration test...")
    config = manager.get_symbol_config('BTC')
    if config:
        print(f"BTC config: {config.to_dict()}")
    
    # ETH有効化テスト
    manager.enable_symbol('ETH')
    print("[OK] ETH enabled")
    
    # 相関行列計算テスト
    correlation_matrix = manager.calculate_correlation_matrix()
    print(f"[OK] Correlation matrix calculated: {correlation_matrix.shape}")
    
    print("\n[SUCCESS] Multi-symbol system test completed!")
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()