#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
デモ用自動ペーパートレーディングテスト
短時間で複数戦略の結果を確認できるデモ
"""

import random
import time
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt

class TradingStrategy:
    """取引戦略クラス"""
    
    def __init__(self, name: str, initial_balance: float = 10000):
        self.name = name
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.trades = []
        self.equity_history = [initial_balance]
        
    def get_signal(self, symbol: str, price: float, price_change: float):
        """戦略に基づくシグナル生成"""
        
        if self.name == 'Conservative':
            # 保守的：大きな変動のみ
            if price_change < -0.03:  # 3%下落で買い
                return 'BUY', 0.8
            elif price_change > 0.04:  # 4%上昇で売り
                return 'SELL', 0.8
                
        elif self.name == 'Aggressive':
            # 積極的：小さな変動でも反応
            if price_change < -0.01:
                return 'BUY', 0.6
            elif price_change > 0.01:
                return 'SELL', 0.6
                
        elif self.name == 'Momentum':
            # モメンタム：トレンドフォロー
            if price_change > 0.02:
                return 'BUY', 0.7
            elif price_change < -0.02:
                return 'SELL', 0.7
                
        elif self.name == 'Contrarian':
            # 逆張り：トレンドに逆らう
            if price_change > 0.025:
                return 'SELL', 0.75
            elif price_change < -0.025:
                return 'BUY', 0.75
                
        elif self.name == 'Balanced':
            # バランス型
            if abs(price_change) > 0.02:
                return 'BUY' if price_change < 0 else 'SELL', 0.65
        
        return 'HOLD', 0.5
    
    def execute_trade(self, symbol: str, action: str, price: float, max_amount: float = 1000):
        """取引実行"""
        fee_rate = 0.001  # 0.1%
        
        if action == 'BUY' and self.balance >= max_amount:
            quantity = max_amount / price
            cost = max_amount + (max_amount * fee_rate)
            
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.trades.append({
                    'action': action,
                    'symbol': symbol,
                    'price': price,
                    'quantity': quantity,
                    'timestamp': datetime.now()
                })
                return True
                
        elif action == 'SELL' and symbol in self.positions and self.positions[symbol] > 0:
            quantity = min(self.positions[symbol], max_amount / price)
            revenue = quantity * price
            fee = revenue * fee_rate
            
            self.balance += revenue - fee
            self.positions[symbol] -= quantity
            self.trades.append({
                'action': action,
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.now()
            })
            return True
            
        return False
    
    def calculate_equity(self, current_prices: dict):
        """現在の総資産計算"""
        equity = self.balance
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                equity += quantity * current_prices[symbol]
        return equity

class DemoAutomatedTest:
    """デモ用自動テストシステム"""
    
    def __init__(self):
        # 5つの戦略を初期化
        self.strategies = {
            'Conservative': TradingStrategy('Conservative'),
            'Aggressive': TradingStrategy('Aggressive'),
            'Momentum': TradingStrategy('Momentum'),
            'Contrarian': TradingStrategy('Contrarian'),
            'Balanced': TradingStrategy('Balanced')
        }
        
        # 初期価格
        self.prices = {
            'BTC': 45000.0,
            'ETH': 3000.0,
            'SOL': 100.0
        }
        
        self.price_history = {symbol: [price] for symbol, price in self.prices.items()}
        self.time_steps = 0
        
    def simulate_market(self):
        """市場シミュレーション"""
        changes = {}
        
        for symbol in self.prices:
            # ランダムウォーク（-5%～+5%）
            change = random.uniform(-0.05, 0.05)
            
            # トレンドを追加（時々）
            if random.random() < 0.1:  # 10%の確率でトレンド
                trend = random.choice([-0.03, 0.03])
                change += trend
            
            # 価格更新
            new_price = self.prices[symbol] * (1 + change)
            self.prices[symbol] = max(new_price, self.prices[symbol] * 0.5)  # 50%以下には下がらない
            
            # 履歴保存
            self.price_history[symbol].append(self.prices[symbol])
            changes[symbol] = change
            
        return changes
    
    def run_test(self, steps: int = 100):
        """テスト実行"""
        print("="*60)
        print("デモ自動ペーパートレーディングテスト")
        print("="*60)
        print(f"テストステップ数: {steps}")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"初期資金: $10,000 (各戦略)")
        print("="*60)
        
        # プログレスバー用
        progress_interval = max(1, steps // 20)
        
        for step in range(steps):
            self.time_steps += 1
            
            # 進捗表示
            if step % progress_interval == 0:
                progress = (step / steps) * 100
                print(f"進捗: {progress:.0f}% (ステップ {step}/{steps})")
            
            # 市場シミュレーション
            price_changes = self.simulate_market()
            
            # 各戦略で取引判断
            for strategy_name, strategy in self.strategies.items():
                for symbol in self.prices:
                    price = self.prices[symbol]
                    change = price_changes[symbol]
                    
                    # シグナル取得
                    action, confidence = strategy.get_signal(symbol, price, change)
                    
                    # 信頼度チェック
                    min_confidence = 0.6
                    if confidence >= min_confidence and action in ['BUY', 'SELL']:
                        # 取引実行
                        trade_amount = 500 + (confidence - 0.5) * 1000  # 信頼度に応じた金額
                        strategy.execute_trade(symbol, action, price, trade_amount)
                
                # エクイティ記録
                equity = strategy.calculate_equity(self.prices)
                strategy.equity_history.append(equity)
        
        # 結果表示
        self.show_results()
        
        # グラフ生成
        self.generate_charts()
    
    def show_results(self):
        """結果表示"""
        print("\n" + "="*60)
        print("テスト結果")
        print("="*60)
        
        results = []
        
        for name, strategy in self.strategies.items():
            final_equity = strategy.calculate_equity(self.prices)
            pnl = final_equity - strategy.initial_balance
            pnl_pct = (pnl / strategy.initial_balance) * 100
            total_trades = len(strategy.trades)
            
            # 勝率計算（簡易版）
            profitable_trades = 0
            for i in range(1, len(strategy.trades)):
                current_trade = strategy.trades[i]
                prev_trade = strategy.trades[i-1]
                
                if (current_trade['action'] == 'SELL' and prev_trade['action'] == 'BUY' and 
                    current_trade['price'] > prev_trade['price']):
                    profitable_trades += 1
                elif (current_trade['action'] == 'BUY' and prev_trade['action'] == 'SELL' and 
                      current_trade['price'] < prev_trade['price']):
                    profitable_trades += 1
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            results.append({
                'strategy': name,
                'final_equity': final_equity,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'total_trades': total_trades,
                'win_rate': win_rate
            })
            
            print(f"\n{name}:")
            print(f"  最終資産: ${final_equity:,.2f}")
            print(f"  損益: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            print(f"  総取引数: {total_trades}")
            print(f"  勝率: {win_rate:.1f}%")
        
        # ランキング
        results.sort(key=lambda x: x['pnl'], reverse=True)
        
        print(f"\n戦略ランキング:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['strategy']}: ${result['pnl']:,.2f} ({result['pnl_pct']:+.1f}%)")
        
        # 結果保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'demo_test_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n結果保存: demo_test_results_{timestamp}.json")
    
    def generate_charts(self):
        """結果のグラフ生成"""
        # 1. エクイティカーブ
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for name, strategy in self.strategies.items():
            plt.plot(strategy.equity_history, label=name, linewidth=2)
        
        plt.title('エクイティカーブ（資産推移）')
        plt.xlabel('時間ステップ')
        plt.ylabel('資産 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 価格チャート
        plt.subplot(2, 2, 2)
        for symbol, prices in self.price_history.items():
            plt.plot(prices, label=symbol, linewidth=2)
        
        plt.title('価格推移')
        plt.xlabel('時間ステップ')
        plt.ylabel('価格 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 最終損益バー
        plt.subplot(2, 2, 3)
        names = list(self.strategies.keys())
        pnls = [self.strategies[name].calculate_equity(self.prices) - 10000 for name in names]
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
        
        plt.bar(names, pnls, color=colors, alpha=0.7)
        plt.title('最終損益')
        plt.xlabel('戦略')
        plt.ylabel('損益 ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. 取引数
        plt.subplot(2, 2, 4)
        trade_counts = [len(self.strategies[name].trades) for name in names]
        plt.bar(names, trade_counts, alpha=0.7)
        plt.title('取引数')
        plt.xlabel('戦略')
        plt.ylabel('取引回数')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'demo_test_charts_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        print(f"グラフ保存: demo_test_charts_{timestamp}.png")
        
        # 表示（オプション）
        # plt.show()
        plt.close()

def main():
    """メイン関数"""
    test = DemoAutomatedTest()
    
    print("デモ自動ペーパートレーディングテストを開始します...")
    print("このテストは約30秒で完了します。")
    
    # 100ステップのテスト実行
    test.run_test(steps=100)
    
    print("\nテスト完了！")

if __name__ == "__main__":
    main()