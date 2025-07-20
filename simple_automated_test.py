#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
シンプルな自動ペーパートレーディングテスト
ルールベースの戦略で1週間のテストを実行
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sqlite3

class SimpleStrategy:
    """シンプルな取引戦略"""
    
    def __init__(self, name: str, config: dict, initial_balance: float = 10000):
        self.name = name
        self.config = config
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.equity_curve = [initial_balance]
        
    def get_signal(self, price_data: dict) -> tuple:
        """取引シグナルを生成"""
        symbol = price_data['symbol']
        price = price_data['price']
        change = price_data.get('change_1h', 0)
        
        # 戦略に応じたシグナル生成
        if self.name == 'conservative':
            # 保守的：大きな下落後に買い、上昇後に売り
            if change < -0.02:  # 2%下落
                return ('BUY', 0.8)  # 80%信頼度
            elif change > 0.03:  # 3%上昇
                return ('SELL', 0.8)
                
        elif self.name == 'aggressive':
            # 積極的：小さな動きでも取引
            if change < -0.01:
                return ('BUY', 0.6)
            elif change > 0.01:
                return ('SELL', 0.6)
                
        elif self.name == 'momentum':
            # モメンタム：トレンドに従う
            if change > 0.015:  # 上昇トレンド
                return ('BUY', 0.7)
            elif change < -0.015:  # 下降トレンド
                return ('SELL', 0.7)
                
        elif self.name == 'contrarian':
            # 逆張り：トレンドに逆らう
            if change > 0.02:  # 上昇時に売り
                return ('SELL', 0.75)
            elif change < -0.02:  # 下落時に買い
                return ('BUY', 0.75)
                
        elif self.name == 'balanced':
            # バランス型：中間的な戦略
            if abs(change) > 0.015:
                if change < 0:
                    return ('BUY', 0.65)
                else:
                    return ('SELL', 0.65)
                    
        return ('HOLD', 0.5)
    
    def execute_trade(self, symbol: str, side: str, price: float, quantity: float):
        """取引を実行"""
        fee = quantity * price * 0.001  # 0.1%手数料
        
        if side == 'BUY':
            cost = quantity * price + fee
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.trades.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'quantity': quantity,
                    'fee': fee
                })
                return True
                
        elif side == 'SELL':
            if self.positions.get(symbol, 0) >= quantity:
                self.balance += quantity * price - fee
                self.positions[symbol] -= quantity
                self.trades.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'quantity': quantity,
                    'fee': fee
                })
                return True
                
        return False
    
    def calculate_equity(self, current_prices: dict) -> float:
        """現在の総資産を計算"""
        equity = self.balance
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                equity += quantity * current_prices[symbol]
        return equity

class SimpleAutomatedTest:
    """シンプルな自動テストシステム"""
    
    def __init__(self):
        self.strategies = {
            'conservative': SimpleStrategy('conservative', {'risk': 0.1}),
            'aggressive': SimpleStrategy('aggressive', {'risk': 0.3}),
            'balanced': SimpleStrategy('balanced', {'risk': 0.2}),
            'momentum': SimpleStrategy('momentum', {'risk': 0.25}),
            'contrarian': SimpleStrategy('contrarian', {'risk': 0.15})
        }
        
        self.current_prices = {
            'BTC': 45000,
            'ETH': 3000,
            'SOL': 100
        }
        
        # データベース初期化
        self.init_database()
        
    def init_database(self):
        """データベース初期化"""
        self.conn = sqlite3.connect('simple_test_results.db')
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                timestamp DATETIME,
                symbol TEXT,
                side TEXT,
                price REAL,
                quantity REAL,
                fee REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                timestamp DATETIME,
                equity REAL,
                balance REAL,
                total_trades INTEGER
            )
        """)
        
        self.conn.commit()
    
    def simulate_price_change(self):
        """価格変動をシミュレート"""
        for symbol in self.current_prices:
            # ランダムな価格変動（-3%～+3%）
            change = random.uniform(-0.03, 0.03)
            self.current_prices[symbol] *= (1 + change)
            
            # 価格データ
            yield {
                'symbol': symbol,
                'price': self.current_prices[symbol],
                'change_1h': change
            }
    
    async def run_test_cycle(self):
        """1サイクルのテスト実行"""
        # 価格変動シミュレーション
        for price_data in self.simulate_price_change():
            symbol = price_data['symbol']
            price = price_data['price']
            
            # 各戦略で判断
            for strategy_name, strategy in self.strategies.items():
                # シグナル取得
                signal, confidence = strategy.get_signal(price_data)
                
                # 信頼度チェック
                min_confidence = 0.6 if strategy_name == 'conservative' else 0.5
                if confidence < min_confidence:
                    continue
                
                # ポジションサイズ計算
                if signal == 'BUY':
                    max_position = strategy.balance * strategy.config['risk']
                    quantity = max_position / price
                    strategy.execute_trade(symbol, 'BUY', price, quantity)
                    
                elif signal == 'SELL' and symbol in strategy.positions:
                    # 保有ポジションの半分を売却
                    quantity = strategy.positions[symbol] * 0.5
                    if quantity > 0:
                        strategy.execute_trade(symbol, 'SELL', price, quantity)
        
        # パフォーマンス記録
        self.record_performance()
    
    def record_performance(self):
        """パフォーマンスを記録"""
        cursor = self.conn.cursor()
        
        for strategy_name, strategy in self.strategies.items():
            equity = strategy.calculate_equity(self.current_prices)
            strategy.equity_curve.append(equity)
            
            cursor.execute("""
                INSERT INTO performance (strategy, timestamp, equity, balance, total_trades)
                VALUES (?, ?, ?, ?, ?)
            """, (
                strategy_name,
                datetime.now(),
                equity,
                strategy.balance,
                len(strategy.trades)
            ))
            
            # 最新の取引を記録
            for trade in strategy.trades[-10:]:  # 最新10件
                if not hasattr(trade, '_recorded'):
                    cursor.execute("""
                        INSERT INTO trades (strategy, timestamp, symbol, side, price, quantity, fee)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        strategy_name,
                        trade['timestamp'],
                        trade['symbol'],
                        trade['side'],
                        trade['price'],
                        trade['quantity'],
                        trade['fee']
                    ))
                    trade['_recorded'] = True
        
        self.conn.commit()
    
    async def run_automated_test(self, duration_hours: int = 168):  # 1週間
        """自動テストを実行"""
        print("="*60)
        print("シンプル自動ペーパートレーディングテスト開始")
        print("="*60)
        print(f"テスト期間: {duration_hours}時間")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        start_time = time.time()
        cycle_count = 0
        
        # テスト実行
        while (time.time() - start_time) < duration_hours * 3600:
            cycle_count += 1
            
            if cycle_count % 10 == 0:  # 10サイクルごとに進捗表示
                elapsed_hours = (time.time() - start_time) / 3600
                progress = (elapsed_hours / duration_hours) * 100
                print(f"\n進捗: {progress:.1f}% ({elapsed_hours:.1f}/{duration_hours}時間)")
                
                # 現在のパフォーマンス表示
                print("\n現在のパフォーマンス:")
                for name, strategy in self.strategies.items():
                    equity = strategy.calculate_equity(self.current_prices)
                    pnl = equity - strategy.initial_balance
                    pnl_pct = (pnl / strategy.initial_balance) * 100
                    print(f"{name}: ${equity:,.2f} (PnL: ${pnl:,.2f}, {pnl_pct:+.1f}%)")
            
            # テストサイクル実行
            await self.run_test_cycle()
            
            # 待機（実際の市場を模擬）
            await asyncio.sleep(1)  # 1秒 = 1時間として高速シミュレーション
        
        # 最終結果
        self.show_final_results()
    
    def show_final_results(self):
        """最終結果を表示"""
        print("\n" + "="*60)
        print("テスト完了 - 最終結果")
        print("="*60)
        
        results = []
        
        for name, strategy in self.strategies.items():
            equity = strategy.calculate_equity(self.current_prices)
            pnl = equity - strategy.initial_balance
            pnl_pct = (pnl / strategy.initial_balance) * 100
            total_trades = len(strategy.trades)
            
            # 勝率計算
            winning_trades = 0
            for i in range(0, len(strategy.trades) - 1, 2):  # Buy-Sellペアで評価
                if i + 1 < len(strategy.trades):
                    buy_trade = strategy.trades[i]
                    sell_trade = strategy.trades[i + 1]
                    if sell_trade['price'] > buy_trade['price']:
                        winning_trades += 1
            
            win_rate = (winning_trades / (total_trades // 2) * 100) if total_trades > 1 else 0
            
            results.append({
                'strategy': name.upper(),
                'final_equity': equity,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'total_trades': total_trades,
                'win_rate': win_rate
            })
            
            print(f"\n{name.upper()}:")
            print(f"  最終資産: ${equity:,.2f}")
            print(f"  損益: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            print(f"  総取引数: {total_trades}")
            print(f"  勝率: {win_rate:.1f}%")
        
        # 最優秀戦略
        best_strategy = max(results, key=lambda x: x['pnl'])
        print(f"\n最優秀戦略: {best_strategy['strategy']}")
        print(f"最高利益: ${best_strategy['pnl']:,.2f} ({best_strategy['pnl_pct']:+.1f}%)")
        
        # 結果をJSONで保存
        with open(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n結果が保存されました。")

async def quick_test():
    """クイックテスト（1時間）"""
    test = SimpleAutomatedTest()
    await test.run_automated_test(duration_hours=1)

async def week_test():
    """1週間テスト"""
    test = SimpleAutomatedTest()
    await test.run_automated_test(duration_hours=168)

async def main():
    """メイン関数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--week':
        await week_test()
    else:
        await quick_test()

if __name__ == "__main__":
    print("シンプル自動テストを開始します...")
    asyncio.run(main())