#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動ペーパートレーディングテストシステム
複数の戦略を自動的にテストし、結果を分析
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from enum import Enum
import sqlite3

# プロジェクトインポート
from core.enhanced_ai_trader import EnhancedAITrader
from core.realistic_paper_trading import RealisticPaperTradingEngine, OrderSide, OrderType
from core.advanced_prediction_engine import AdvancedPredictionEngine
from core.risk_management_system import RiskManagementSystem

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingStrategy(Enum):
    """取引戦略の定義"""
    CONSERVATIVE = "conservative"      # 保守的：高信頼度のみ
    AGGRESSIVE = "aggressive"          # 積極的：低信頼度でも取引
    BALANCED = "balanced"              # バランス型
    MOMENTUM = "momentum"              # モメンタム追従
    CONTRARIAN = "contrarian"          # 逆張り
    AI_ONLY = "ai_only"               # AI予測のみ
    HYBRID = "hybrid"                  # AI+テクニカル

class AutomatedPaperTestingSystem:
    """自動ペーパートレーディングテストシステム"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.strategies = {}
        self.test_results = {}
        self.is_running = False
        
        # データベース初期化
        self.db_path = Path("paper_test_results.db")
        self._init_database()
        
        # 戦略パラメータ
        self.strategy_configs = {
            TradingStrategy.CONSERVATIVE: {
                "min_confidence": 0.8,
                "max_position_pct": 0.1,  # 10%
                "stop_loss": 0.03,        # 3%
                "take_profit": 0.05,      # 5%
                "max_trades_per_day": 3
            },
            TradingStrategy.AGGRESSIVE: {
                "min_confidence": 0.5,
                "max_position_pct": 0.3,  # 30%
                "stop_loss": 0.05,        # 5%
                "take_profit": 0.10,      # 10%
                "max_trades_per_day": 10
            },
            TradingStrategy.BALANCED: {
                "min_confidence": 0.65,
                "max_position_pct": 0.2,  # 20%
                "stop_loss": 0.04,        # 4%
                "take_profit": 0.07,      # 7%
                "max_trades_per_day": 5
            },
            TradingStrategy.MOMENTUM: {
                "min_confidence": 0.6,
                "max_position_pct": 0.25,
                "stop_loss": 0.04,
                "take_profit": 0.08,
                "max_trades_per_day": 7,
                "trend_following": True
            },
            TradingStrategy.CONTRARIAN: {
                "min_confidence": 0.7,
                "max_position_pct": 0.15,
                "stop_loss": 0.03,
                "take_profit": 0.06,
                "max_trades_per_day": 5,
                "reverse_signals": True
            },
            TradingStrategy.AI_ONLY: {
                "min_confidence": 0.6,
                "max_position_pct": 0.2,
                "stop_loss": 0.04,
                "take_profit": 0.08,
                "max_trades_per_day": 6,
                "use_ai_only": True
            },
            TradingStrategy.HYBRID: {
                "min_confidence": 0.65,
                "max_position_pct": 0.2,
                "stop_loss": 0.035,
                "take_profit": 0.07,
                "max_trades_per_day": 8,
                "use_technical": True
            }
        }
        
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 取引履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                confidence REAL,
                signal TEXT
            )
        """)
        
        # パフォーマンス統計テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                date DATE NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def initialize_strategies(self):
        """戦略の初期化"""
        logger.info("戦略を初期化中...")
        
        for strategy in TradingStrategy:
            try:
                # 各戦略用のトレーダーを作成
                trader = EnhancedAITrader(self.initial_balance)
                realistic_trader = RealisticPaperTradingEngine(self.initial_balance)
                
                self.strategies[strategy] = {
                    'trader': trader,
                    'realistic_trader': realistic_trader,
                    'config': self.strategy_configs[strategy],
                    'daily_trades': 0,
                    'last_trade_time': None,
                    'performance': {
                        'trades': [],
                        'daily_pnl': [],
                        'equity_curve': [self.initial_balance]
                    }
                }
                
                logger.info(f"戦略 {strategy.value} を初期化しました")
                
            except Exception as e:
                logger.error(f"戦略 {strategy.value} の初期化エラー: {e}")
    
    async def execute_strategy(self, strategy: TradingStrategy, symbol: str = 'BTC'):
        """戦略の実行"""
        strategy_data = self.strategies[strategy]
        config = strategy_data['config']
        trader = strategy_data['trader']
        
        try:
            # AI予測を取得
            prediction = trader.get_enhanced_prediction(symbol)
            confidence = prediction.get('confidence', 0)
            signal = prediction.get('signal', 'HOLD')
            
            # 戦略固有の条件チェック
            if not self._should_trade(strategy, confidence, signal):
                return
            
            # 日次取引制限チェック
            if strategy_data['daily_trades'] >= config['max_trades_per_day']:
                logger.info(f"{strategy.value}: 日次取引制限に達しました")
                return
            
            # ポジションサイズ計算
            account = trader.trading_engine.get_account_summary()
            position_size = self._calculate_position_size(
                strategy, account['equity'], confidence
            )
            
            if position_size <= 0:
                return
            
            # 注文実行
            order_side = self._determine_order_side(strategy, signal)
            if order_side:
                success = await self._execute_order(
                    strategy, symbol, order_side, position_size
                )
                
                if success:
                    strategy_data['daily_trades'] += 1
                    strategy_data['last_trade_time'] = datetime.now()
                    logger.info(f"{strategy.value}: {order_side} 注文実行 - {symbol} ${position_size:.2f}")
            
        except Exception as e:
            logger.error(f"戦略実行エラー {strategy.value}: {e}")
    
    def _should_trade(self, strategy: TradingStrategy, confidence: float, signal: str) -> bool:
        """取引すべきかの判断"""
        config = self.strategy_configs[strategy]
        
        # 信頼度チェック
        if confidence < config['min_confidence']:
            return False
        
        # HOLDシグナルは基本的に取引しない
        if signal == 'HOLD':
            return False
        
        # 戦略固有の条件
        if strategy == TradingStrategy.CONTRARIAN and config.get('reverse_signals'):
            # 逆張り戦略では信号を反転
            return True
        
        return True
    
    def _calculate_position_size(self, strategy: TradingStrategy, equity: float, confidence: float) -> float:
        """ポジションサイズの計算"""
        config = self.strategy_configs[strategy]
        max_position = equity * config['max_position_pct']
        
        # 信頼度に応じてサイズ調整
        confidence_factor = (confidence - config['min_confidence']) / (1.0 - config['min_confidence'])
        position_size = max_position * confidence_factor
        
        # 最小取引サイズ
        min_size = 100  # $100
        if position_size < min_size:
            return 0
        
        return min(position_size, max_position)
    
    def _determine_order_side(self, strategy: TradingStrategy, signal: str) -> Optional[OrderSide]:
        """注文方向の決定"""
        config = self.strategy_configs[strategy]
        
        if config.get('reverse_signals'):
            # 逆張り戦略
            if signal == 'BUY':
                return OrderSide.SELL
            elif signal == 'SELL':
                return OrderSide.BUY
        else:
            # 通常戦略
            if signal == 'BUY':
                return OrderSide.BUY
            elif signal == 'SELL':
                return OrderSide.SELL
        
        return None
    
    async def _execute_order(self, strategy: TradingStrategy, symbol: str, 
                           side: OrderSide, amount: float) -> bool:
        """注文の実行"""
        strategy_data = self.strategies[strategy]
        trader = strategy_data['realistic_trader']
        
        try:
            # 成行注文実行
            order_id = trader.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=amount / trader.get_current_price(symbol)  # 金額から数量に変換
            )
            
            if order_id:
                # 取引記録
                self._record_trade(strategy, symbol, side, amount)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"注文実行エラー: {e}")
            return False
    
    def _record_trade(self, strategy: TradingStrategy, symbol: str, side: OrderSide, amount: float):
        """取引の記録"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (strategy, symbol, side, quantity, price, confidence, signal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.value,
            symbol,
            side.value,
            amount,
            self.strategies[strategy]['realistic_trader'].get_current_price(symbol),
            0.0,  # 後で更新
            'AI'
        ))
        
        conn.commit()
        conn.close()
    
    async def run_test_cycle(self):
        """テストサイクルの実行"""
        symbols = ['BTC', 'ETH', 'SOL']
        
        while self.is_running:
            try:
                # 各戦略を実行
                for strategy in TradingStrategy:
                    for symbol in symbols:
                        await self.execute_strategy(strategy, symbol)
                
                # ポジション管理（ストップロス、テイクプロフィット）
                await self._manage_positions()
                
                # パフォーマンス更新
                self._update_performance()
                
                # 次のサイクルまで待機（5分）
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"テストサイクルエラー: {e}")
                await asyncio.sleep(60)
    
    async def _manage_positions(self):
        """ポジション管理（ストップロス、テイクプロフィット）"""
        for strategy in TradingStrategy:
            strategy_data = self.strategies[strategy]
            trader = strategy_data['realistic_trader']
            config = strategy_data['config']
            
            positions = trader.get_positions()
            
            for symbol, position in positions.items():
                if position['quantity'] == 0:
                    continue
                
                current_price = trader.get_current_price(symbol)
                entry_price = position['avg_entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                # ストップロスチェック
                if pnl_pct <= -config['stop_loss']:
                    logger.info(f"{strategy.value}: ストップロス発動 {symbol}")
                    await self._execute_order(
                        strategy, symbol, 
                        OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY,
                        abs(position['quantity'] * current_price)
                    )
                
                # テイクプロフィットチェック
                elif pnl_pct >= config['take_profit']:
                    logger.info(f"{strategy.value}: テイクプロフィット発動 {symbol}")
                    await self._execute_order(
                        strategy, symbol,
                        OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY,
                        abs(position['quantity'] * current_price)
                    )
    
    def _update_performance(self):
        """パフォーマンスの更新"""
        for strategy in TradingStrategy:
            strategy_data = self.strategies[strategy]
            trader = strategy_data['realistic_trader']
            
            # アカウントサマリー取得
            account = trader.get_account_summary()
            equity = account['equity']
            
            # エクイティカーブ更新
            strategy_data['performance']['equity_curve'].append(equity)
            
            # 日次パフォーマンス計算
            if len(strategy_data['performance']['equity_curve']) > 1:
                daily_return = (equity - strategy_data['performance']['equity_curve'][-2]) / strategy_data['performance']['equity_curve'][-2]
                strategy_data['performance']['daily_pnl'].append(daily_return)
    
    def _reset_daily_counters(self):
        """日次カウンターのリセット"""
        for strategy in self.strategies.values():
            strategy['daily_trades'] = 0
    
    async def start_automated_test(self, duration_days: int = 7):
        """自動テストの開始"""
        logger.info(f"自動ペーパートレーディングテスト開始 - 期間: {duration_days}日")
        
        self.is_running = True
        self.initialize_strategies()
        
        # テスト終了時刻
        end_time = datetime.now() + timedelta(days=duration_days)
        
        # 日次リセットタスク
        async def daily_reset():
            while self.is_running:
                await asyncio.sleep(86400)  # 24時間
                self._reset_daily_counters()
                self._save_daily_performance()
        
        # タスク並列実行
        tasks = [
            asyncio.create_task(self.run_test_cycle()),
            asyncio.create_task(daily_reset())
        ]
        
        # 終了時刻まで実行
        while datetime.now() < end_time and self.is_running:
            await asyncio.sleep(60)
        
        self.is_running = False
        
        # タスクキャンセル
        for task in tasks:
            task.cancel()
        
        # 最終レポート生成
        report = self.generate_final_report()
        self._save_report(report)
        
        logger.info("自動テスト完了")
        return report
    
    def _save_daily_performance(self):
        """日次パフォーマンスの保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for strategy in TradingStrategy:
            strategy_data = self.strategies[strategy]
            performance = self._calculate_performance_metrics(strategy)
            
            cursor.execute("""
                INSERT INTO performance 
                (strategy, date, total_trades, winning_trades, total_pnl, 
                 max_drawdown, sharpe_ratio, win_rate, avg_win, avg_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.value,
                datetime.now().date(),
                performance['total_trades'],
                performance['winning_trades'],
                performance['total_pnl'],
                performance['max_drawdown'],
                performance['sharpe_ratio'],
                performance['win_rate'],
                performance['avg_win'],
                performance['avg_loss']
            ))
        
        conn.commit()
        conn.close()
    
    def _calculate_performance_metrics(self, strategy: TradingStrategy) -> Dict:
        """パフォーマンス指標の計算"""
        strategy_data = self.strategies[strategy]
        equity_curve = strategy_data['performance']['equity_curve']
        
        if len(equity_curve) < 2:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        # 総損益
        total_pnl = equity_curve[-1] - self.initial_balance
        
        # 最大ドローダウン
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # シャープレシオ（簡易版）
        if len(strategy_data['performance']['daily_pnl']) > 0:
            returns = np.array(strategy_data['performance']['daily_pnl'])
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 取引統計（データベースから取得）
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                   AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
            FROM trades
            WHERE strategy = ?
        """, (strategy.value,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_trades': result[0] or 0,
            'winning_trades': result[1] or 0,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': (result[1] / result[0] * 100) if result[0] > 0 else 0,
            'avg_win': result[2] or 0,
            'avg_loss': result[3] or 0
        }
    
    def generate_final_report(self) -> Dict:
        """最終レポートの生成"""
        report = {
            'test_duration': '7 days',
            'strategies': {},
            'best_strategy': None,
            'summary': {}
        }
        
        best_pnl = -float('inf')
        best_strategy = None
        
        for strategy in TradingStrategy:
            metrics = self._calculate_performance_metrics(strategy)
            report['strategies'][strategy.value] = metrics
            
            if metrics['total_pnl'] > best_pnl:
                best_pnl = metrics['total_pnl']
                best_strategy = strategy.value
        
        report['best_strategy'] = best_strategy
        report['summary'] = {
            'total_strategies_tested': len(TradingStrategy),
            'best_total_pnl': best_pnl,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _save_report(self, report: Dict):
        """レポートの保存"""
        report_path = Path(f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"レポート保存完了: {report_path}")

# テスト実行用関数
async def run_automated_test():
    """自動テストの実行"""
    system = AutomatedPaperTestingSystem(initial_balance=10000)
    report = await system.start_automated_test(duration_days=7)
    
    print("\n=== 自動テスト完了 ===")
    print(f"最良戦略: {report['best_strategy']}")
    print(f"最高利益: ${report['summary']['best_total_pnl']:.2f}")
    
    return report

if __name__ == "__main__":
    # テスト実行
    asyncio.run(run_automated_test())