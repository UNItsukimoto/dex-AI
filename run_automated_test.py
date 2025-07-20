#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動ペーパートレーディングテスト実行スクリプト
1週間の自動テストを簡単に開始できる
"""

import asyncio
import sys
import os
from datetime import datetime
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.automated_paper_testing import AutomatedPaperTestingSystem, TradingStrategy
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutomatedTestRunner:
    """自動テストランナー"""
    
    def __init__(self, initial_balance: float = 10000, test_duration: int = 7):
        self.initial_balance = initial_balance
        self.test_duration = test_duration
        self.test_system = AutomatedPaperTestingSystem(initial_balance)
        
    async def run_full_test(self):
        """完全な自動テストを実行"""
        print("="*60)
        print("自動ペーパートレーディングテスト開始")
        print("="*60)
        print(f"初期資金: ${self.initial_balance:,.2f}")
        print(f"テスト期間: {self.test_duration}日間")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # テスト戦略の表示
        print("\nテストする戦略:")
        for i, strategy in enumerate(TradingStrategy, 1):
            config = self.test_system.strategy_configs[strategy]
            print(f"{i}. {strategy.value.upper()}")
            print(f"   - 最小信頼度: {config['min_confidence']*100:.0f}%")
            print(f"   - 最大ポジション: {config['max_position_pct']*100:.0f}%")
            print(f"   - ストップロス: {config['stop_loss']*100:.1f}%")
            print(f"   - テイクプロフィット: {config['take_profit']*100:.1f}%")
        
        print("\nテスト実行中...")
        
        # 自動テスト実行
        try:
            report = await self.test_system.start_automated_test(self.test_duration)
            
            # 結果表示
            self.display_results(report)
            
            # グラフ生成
            self.generate_charts(report)
            
            return report
            
        except Exception as e:
            logger.error(f"テスト実行エラー: {e}")
            raise
    
    def display_results(self, report: dict):
        """テスト結果の表示"""
        print("\n" + "="*60)
        print("テスト結果サマリー")
        print("="*60)
        
        # 戦略別結果
        results_data = []
        for strategy_name, metrics in report['strategies'].items():
            results_data.append({
                '戦略': strategy_name.upper(),
                '総取引数': metrics['total_trades'],
                '勝率': f"{metrics['win_rate']:.1f}%",
                '総損益': f"${metrics['total_pnl']:,.2f}",
                '最大DD': f"{metrics['max_drawdown']*100:.1f}%",
                'シャープ比': f"{metrics['sharpe_ratio']:.2f}"
            })
        
        # DataFrame表示
        df = pd.DataFrame(results_data)
        print(df.to_string(index=False))
        
        print(f"\n最優秀戦略: {report['best_strategy'].upper()}")
        print(f"最高利益: ${report['summary']['best_total_pnl']:,.2f}")
    
    def generate_charts(self, report: dict):
        """結果のグラフ生成"""
        # 保存ディレクトリ
        charts_dir = Path("test_results_charts")
        charts_dir.mkdir(exist_ok=True)
        
        # スタイル設定
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 戦略別損益比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategies = list(report['strategies'].keys())
        pnls = [report['strategies'][s]['total_pnl'] for s in strategies]
        win_rates = [report['strategies'][s]['win_rate'] for s in strategies]
        
        # 損益棒グラフ
        bars = ax1.bar(strategies, pnls)
        for i, bar in enumerate(bars):
            if pnls[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax1.set_title('戦略別総損益', fontsize=14)
        ax1.set_xlabel('戦略')
        ax1.set_ylabel('損益 ($)')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 勝率散布図
        ax2.scatter(win_rates, pnls, s=100)
        for i, strategy in enumerate(strategies):
            ax2.annotate(strategy, (win_rates[i], pnls[i]), fontsize=8)
        
        ax2.set_title('勝率 vs 損益', fontsize=14)
        ax2.set_xlabel('勝率 (%)')
        ax2.set_ylabel('総損益 ($)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=50, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. リスク・リターン分析
        fig, ax = plt.subplots(figsize=(10, 8))
        
        max_dds = [report['strategies'][s]['max_drawdown']*100 for s in strategies]
        returns = [(pnl/self.initial_balance)*100 for pnl in pnls]
        
        scatter = ax.scatter(max_dds, returns, s=200, alpha=0.7)
        
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy.upper(), (max_dds[i], returns[i]), 
                       fontsize=10, ha='center', va='bottom')
        
        ax.set_title('リスク・リターン分析', fontsize=16)
        ax.set_xlabel('最大ドローダウン (%)', fontsize=12)
        ax.set_ylabel('リターン (%)', fontsize=12)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 理想的な領域を強調
        ax.axvspan(0, 5, ymin=0.5, alpha=0.1, color='green', label='理想的')
        
        plt.savefig(charts_dir / 'risk_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nグラフ保存完了: {charts_dir}")

async def run_quick_test():
    """クイックテスト（1日）"""
    runner = AutomatedTestRunner(initial_balance=10000, test_duration=1)
    await runner.run_full_test()

async def run_week_test():
    """1週間テスト"""
    runner = AutomatedTestRunner(initial_balance=10000, test_duration=7)
    await runner.run_full_test()

async def run_custom_test(balance: float, days: int):
    """カスタムテスト"""
    runner = AutomatedTestRunner(initial_balance=balance, test_duration=days)
    await runner.run_full_test()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='自動ペーパートレーディングテスト')
    parser.add_argument('--mode', choices=['quick', 'week', 'custom'], 
                       default='week', help='テストモード')
    parser.add_argument('--balance', type=float, default=10000, 
                       help='初期資金（カスタムモード用）')
    parser.add_argument('--days', type=int, default=7, 
                       help='テスト日数（カスタムモード用）')
    
    args = parser.parse_args()
    
    # 非同期実行
    if args.mode == 'quick':
        asyncio.run(run_quick_test())
    elif args.mode == 'week':
        asyncio.run(run_week_test())
    else:
        asyncio.run(run_custom_test(args.balance, args.days))

if __name__ == "__main__":
    main()