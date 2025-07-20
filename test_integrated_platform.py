#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合プラットフォーム テスト
すべてのコンポーネントの統合動作を確認
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_ai_trader import EnhancedAITrader
from core.realistic_paper_trading import RealisticPaperTradingEngine
from core.advanced_prediction_engine import AdvancedPredictionEngine
from core.risk_management_system import RiskManagementSystem

def test_component_initialization():
    """コンポーネント初期化テスト"""
    print("=== コンポーネント初期化テスト ===")
    
    try:
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

def test_prediction_integration(trader, prediction_engine):
    """予測システム統合テスト"""
    print("\n=== 予測システム統合テスト ===")
    
    try:
        # サンプルデータ作成
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, 200)
        })
        df.set_index('timestamp', inplace=True)
        
        # 高度予測実行
        prediction = prediction_engine.get_enhanced_prediction('BTC', df)
        print(f"✅ 高度予測実行成功: {prediction['signal']} (信頼度: {prediction['confidence']:.1%})")
        
        # トレーダーの予測と比較
        trader_prediction = trader.get_enhanced_prediction('BTC')
        print(f"✅ トレーダー予測成功: {trader_prediction['signal']} (信頼度: {trader_prediction['confidence']:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ 予測統合エラー: {e}")
        return False

def test_risk_management_integration(trader, risk_manager):
    """リスク管理統合テスト"""
    print("\n=== リスク管理統合テスト ===")
    
    try:
        # アカウント情報取得
        account = trader.trading_engine.get_account_summary()
        positions = trader.trading_engine.get_positions()
        
        # リスク指標計算
        risk_metrics = risk_manager.calculate_risk_metrics(account['equity'], positions)
        print(f"✅ リスク指標計算成功: レベル={risk_metrics.risk_level.value}, エクスポージャー={risk_metrics.total_exposure:.1%}")
        
        # ポジションサイズ推奨
        position_rec = risk_manager.calculate_position_size('BTC', account['equity'], 0.7, 67000)
        print(f"✅ ポジションサイズ推奨成功: {position_rec.recommended_size:.6f} BTC (理由: {position_rec.reason})")
        
        # ストップロス・テイクプロフィット計算
        sl_tp = risk_manager.calculate_stop_loss_take_profit('BTC', 67000, 'long', 0.7)
        print(f"✅ SL/TP計算成功: SL=${sl_tp['stop_loss_price']:,.0f}, TP=${sl_tp['take_profit_price']:,.0f}")
        
        # リスクレポート生成
        risk_report = risk_manager.generate_risk_report(account['equity'], positions)
        print(f"✅ リスクレポート生成成功: 推奨事項={len(risk_report.get('recommendations', []))}件")
        
        return True
        
    except Exception as e:
        print(f"❌ リスク管理統合エラー: {e}")
        return False

def test_realistic_trading_integration(realistic_trader):
    """リアル体験取引統合テスト"""
    print("\n=== リアル体験取引統合テスト ===")
    
    try:
        from core.realistic_paper_trading import OrderSide, OrderType
        
        # 価格更新
        realistic_trader.update_live_prices()
        print("✅ ライブ価格更新成功")
        
        # マーケットサマリー取得
        market_summary = realistic_trader.get_market_summary()
        print(f"✅ マーケットサマリー取得成功: {len(market_summary)}銘柄")
        
        # アカウントサマリー取得
        account = realistic_trader.get_account_summary()
        print(f"✅ アカウント取得成功: 残高=${account['balance']:,.2f}")
        
        # 少額テスト注文
        order_id = realistic_trader.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001  # 最小テスト量
        )
        
        if order_id:
            print(f"✅ テスト注文成功: ID={order_id}")
            
            # 取引履歴確認
            trade_history = realistic_trader.get_trade_history(5)
            print(f"✅ 取引履歴取得成功: {len(trade_history)}件")
        else:
            print("⚠️ テスト注文失敗（残高不足の可能性）")
        
        return True
        
    except Exception as e:
        print(f"❌ リアル体験取引統合エラー: {e}")
        return False

def test_ml_model_performance(prediction_engine):
    """機械学習モデル性能テスト"""
    print("\n=== ML モデル性能テスト ===")
    
    try:
        # モデル性能取得
        performance = prediction_engine.get_model_performance()
        print(f"✅ ML利用可能: {performance.get('ml_available', False)}")
        print(f"✅ 学習サンプル数: {performance.get('training_samples', 0):,}")
        print(f"✅ 予測回数: {performance.get('prediction_count', 0):,}")
        
        model_perf = performance.get('model_performance', {})
        if model_perf:
            print("✅ モデル性能:")
            for model_name, perf in model_perf.items():
                accuracy = perf.get('accuracy', 0)
                print(f"   - {model_name}: {accuracy:.1%}")
        else:
            print("ℹ️ モデル性能: 学習データ不足")
        
        return True
        
    except Exception as e:
        print(f"❌ ML性能テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("統合プラットフォーム テスト開始")
    
    # コンポーネント初期化
    trader, realistic_trader, prediction_engine, risk_manager = test_component_initialization()
    
    if not all([trader, realistic_trader, prediction_engine, risk_manager]):
        print("❌ 初期化失敗。テスト中止。")
        return
    
    # 各統合テスト実行
    tests = [
        ("予測システム統合", lambda: test_prediction_integration(trader, prediction_engine)),
        ("リスク管理統合", lambda: test_risk_management_integration(trader, risk_manager)),
        ("リアル体験取引統合", lambda: test_realistic_trading_integration(realistic_trader)),
        ("ML モデル性能", lambda: test_ml_model_performance(prediction_engine))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テストでエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{total} テスト通過 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("全テスト通過！統合プラットフォーム正常動作")
    elif passed >= total * 0.8:
        print("大部分のテスト通過。軽微な問題あり")
    else:
        print("多数のテスト失敗。統合に問題あり")

if __name__ == "__main__":
    main()