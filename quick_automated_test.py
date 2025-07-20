#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
クイック自動テスト（MLモデル学習を含む）
初期データで学習してから自動テストを実行
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.automated_paper_testing import AutomatedPaperTestingSystem, TradingStrategy
from core.advanced_prediction_engine import AdvancedPredictionEngine

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickAutomatedTest:
    """クイック自動テスト"""
    
    def __init__(self):
        self.test_system = None
        
    def prepare_training_data(self) -> pd.DataFrame:
        """学習用データの準備"""
        # サンプルデータ生成（200時間分）
        dates = pd.date_range(end=datetime.now(), periods=200, freq='H')
        
        # BTCの価格データをシミュレート
        np.random.seed(42)
        price = 50000
        prices = []
        
        for _ in range(200):
            change = np.random.normal(0, 0.01)  # 1%のボラティリティ
            price = price * (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.lognormal(10, 0.5, 200)
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def run_quick_test(self):
        """クイックテスト実行"""
        print("="*60)
        print("クイック自動ペーパートレーディングテスト")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: 学習データ準備
        print("\n[Step 1] 学習データ準備中...")
        training_data = {
            'BTC': self.prepare_training_data(),
            'ETH': self.prepare_training_data(),
            'SOL': self.prepare_training_data()
        }
        
        # Step 2: 各戦略のモデルを学習
        print("\n[Step 2] 各戦略のMLモデルを学習中...")
        self.test_system = AutomatedPaperTestingSystem(initial_balance=10000)
        self.test_system.initialize_strategies()
        
        for strategy in TradingStrategy:
            print(f"  - {strategy.value} モデル学習中...")
            trader = self.test_system.strategies[strategy]['trader']
            
            # 各銘柄でモデルを学習
            for symbol, data in training_data.items():
                try:
                    # 予測エンジンに学習データを設定
                    trader.prediction_engine.train_models(symbol, data)
                except Exception as e:
                    logger.warning(f"学習スキップ {strategy.value} - {symbol}: {e}")
        
        print("\n[Step 3] テスト実行中...")
        
        # 短時間のテスト実行（10分）
        test_duration_minutes = 10
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=test_duration_minutes)
        
        cycle_count = 0
        while datetime.now() < end_time:
            cycle_count += 1
            print(f"\nサイクル {cycle_count} 実行中...")
            
            # 各戦略を実行
            for strategy in TradingStrategy:
                for symbol in ['BTC', 'ETH', 'SOL']:
                    try:
                        await self.test_system.execute_strategy(strategy, symbol)
                    except Exception as e:
                        logger.error(f"戦略実行エラー {strategy.value} - {symbol}: {e}")
            
            # ポジション管理
            await self.test_system._manage_positions()
            
            # パフォーマンス更新
            self.test_system._update_performance()
            
            # 30秒待機
            await asyncio.sleep(30)
        
        # Step 4: 結果生成
        print("\n[Step 4] 結果分析中...")
        report = self.test_system.generate_final_report()
        
        # 結果表示
        self.display_results(report)
        
        return report
    
    def display_results(self, report: dict):
        """結果表示"""
        print("\n" + "="*60)
        print("テスト結果")
        print("="*60)
        
        # 戦略別結果
        print("\n戦略別パフォーマンス:")
        print("-"*40)
        
        for strategy_name, metrics in report['strategies'].items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  総取引数: {metrics['total_trades']}")
            print(f"  勝率: {metrics['win_rate']:.1f}%")
            print(f"  総損益: ${metrics['total_pnl']:,.2f}")
            print(f"  最大DD: {metrics['max_drawdown']*100:.1f}%")
            print(f"  シャープ比: {metrics['sharpe_ratio']:.2f}")
        
        print(f"\n最優秀戦略: {report['best_strategy'].upper()}")
        print(f"最高利益: ${report['summary']['best_total_pnl']:,.2f}")
        
        # データベース保存
        print(f"\n結果はデータベースに保存されました: paper_test_results.db")

async def main():
    """メイン関数"""
    test = QuickAutomatedTest()
    
    try:
        await test.run_quick_test()
    except KeyboardInterrupt:
        print("\nテスト中断")
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        raise

if __name__ == "__main__":
    print("クイック自動テストを開始します...")
    print("このテストは約10分で完了します。")
    print("Ctrl+C で中断できます。")
    print("")
    
    asyncio.run(main())