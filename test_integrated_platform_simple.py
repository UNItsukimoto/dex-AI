#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合プラットフォーム 簡易テスト
文字エンコーディング問題を回避した簡易版
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """インポートテスト"""
    print("=== インポートテスト ===")
    
    try:
        from core.enhanced_ai_trader import EnhancedAITrader
        print("[OK] EnhancedAITrader インポート成功")
        
        from core.realistic_paper_trading import RealisticPaperTradingEngine
        print("[OK] RealisticPaperTradingEngine インポート成功")
        
        from core.advanced_prediction_engine import AdvancedPredictionEngine
        print("[OK] AdvancedPredictionEngine インポート成功")
        
        from core.risk_management_system import RiskManagementSystem
        print("[OK] RiskManagementSystem インポート成功")
        
        return True
    except Exception as e:
        print(f"[ERROR] インポートエラー: {e}")
        return False

def test_initialization():
    """初期化テスト"""
    print("\n=== 初期化テスト ===")
    
    try:
        from core.enhanced_ai_trader import EnhancedAITrader
        from core.realistic_paper_trading import RealisticPaperTradingEngine
        from core.advanced_prediction_engine import AdvancedPredictionEngine
        from core.risk_management_system import RiskManagementSystem
        
        # 強化AIトレーダー
        trader = EnhancedAITrader(10000)
        print("[OK] 強化AIトレーダー初期化成功")
        
        # リアル体験トレーダー
        realistic_trader = RealisticPaperTradingEngine(10000)
        print("[OK] リアル体験トレーダー初期化成功")
        
        # 高度予測エンジン
        prediction_engine = AdvancedPredictionEngine()
        print("[OK] 高度予測エンジン初期化成功")
        
        # リスク管理システム
        risk_manager = RiskManagementSystem()
        print("[OK] リスク管理システム初期化成功")
        
        return trader, realistic_trader, prediction_engine, risk_manager
        
    except Exception as e:
        print(f"[ERROR] 初期化エラー: {e}")
        return None, None, None, None

def test_basic_functionality(trader, realistic_trader, prediction_engine, risk_manager):
    """基本機能テスト"""
    print("\n=== 基本機能テスト ===")
    
    try:
        # アカウント情報取得
        account = trader.trading_engine.get_account_summary()
        print(f"[OK] アカウント取得: 残高=${account['balance']:,.2f}")
        
        # 予測エンジン性能取得
        ml_performance = prediction_engine.get_model_performance()
        print(f"[OK] ML性能取得: 利用可能={ml_performance.get('ml_available', False)}")
        
        # リスク指標計算
        positions = trader.trading_engine.get_positions()
        risk_metrics = risk_manager.calculate_risk_metrics(account['equity'], positions)
        print(f"[OK] リスク指標計算: レベル={risk_metrics.risk_level.value}")
        
        # マーケットデータ取得
        market_summary = realistic_trader.get_market_summary()
        print(f"[OK] マーケットデータ取得: {len(market_summary)}銘柄")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 基本機能テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("統合プラットフォーム 簡易テスト開始")
    
    # テスト実行
    results = []
    
    # インポートテスト
    import_result = test_imports()
    results.append(("インポート", import_result))
    
    if not import_result:
        print("[ERROR] インポート失敗。テスト中止。")
        return
    
    # 初期化テスト
    trader, realistic_trader, prediction_engine, risk_manager = test_initialization()
    init_result = all([trader, realistic_trader, prediction_engine, risk_manager])
    results.append(("初期化", init_result))
    
    if not init_result:
        print("[ERROR] 初期化失敗。テスト中止。")
        return
    
    # 基本機能テスト
    func_result = test_basic_functionality(trader, realistic_trader, prediction_engine, risk_manager)
    results.append(("基本機能", func_result))
    
    # 結果サマリー
    print("\n" + "="*40)
    print("テスト結果サマリー")
    print("="*40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{total} テスト通過 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] 全テスト通過！統合プラットフォーム正常動作")
    elif passed >= total * 0.8:
        print("[WARNING] 大部分のテスト通過。軽微な問題あり")
    else:
        print("[ERROR] 多数のテスト失敗。統合に問題あり")

if __name__ == "__main__":
    main()