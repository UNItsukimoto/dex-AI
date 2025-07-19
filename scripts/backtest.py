#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテストスクリプト
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)

class Backtester:
    """バックテストクラス"""
    
    def __init__(self, initial_capital: float = 10000, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reset()
        
    def reset(self):
        """状態をリセット"""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
    def buy(self, price: float, size: float = None):
        """買い注文"""
        if size is None:
            size = self.capital / price * (1 - self.fee_rate)
        
        cost = size * price * (1 + self.fee_rate)
        
        if cost <= self.capital:
            self.capital -= cost
            self.position += size
            self.trades.append({
                'type': 'buy',
                'price': price,
                'size': size,
                'cost': cost,
                'timestamp': datetime.now()
            })
            logger.info(f"Buy: {size:.4f} @ ${price:.2f}")
    
    def sell(self, price: float, size: float = None):
        """売り注文"""
        if size is None:
            size = self.position
        
        if size <= self.position:
            revenue = size * price * (1 - self.fee_rate)
            self.capital += revenue
            self.position -= size
            self.trades.append({
                'type': 'sell',
                'price': price,
                'size': size,
                'revenue': revenue,
                'timestamp': datetime.now()
            })
            logger.info(f"Sell: {size:.4f} @ ${price:.2f}")
    
    def run_backtest(self, prices: np.ndarray, predictions: np.ndarray, 
                    strategy: str = 'threshold'):
        """バックテストを実行"""
        self.reset()
        
        for i in range(len(predictions)):
            current_price = prices[i]
            predicted_price = predictions[i]
            
            # 現在の資産価値
            current_equity = self.capital + self.position * current_price
            self.equity_curve.append(current_equity)
            
            # 戦略に基づく取引
            if strategy == 'threshold':
                self._threshold_strategy(current_price, predicted_price)
            elif strategy == 'ml_confidence':
                self._ml_confidence_strategy(current_price, predicted_price, i)
    
    def _threshold_strategy(self, current_price: float, predicted_price: float, 
                          threshold: float = 0.01):
        """閾値ベースの戦略"""
        predicted_change = (predicted_price - current_price) / current_price
        
        if predicted_change > threshold and self.position == 0:
            # 買いシグナル
            self.buy(current_price)
        elif predicted_change < -threshold and self.position > 0:
            # 売りシグナル
            self.sell(current_price)
    
    def calculate_metrics(self) -> dict:
        """パフォーマンスメトリクスを計算"""
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # 総リターン
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # シャープレシオ（年率換算）
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # 最大ドローダウン
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # 勝率
        profitable_trades = [t for t in self.trades if t['type'] == 'sell' 
                           and t.get('revenue', 0) > t.get('cost', 0)]
        win_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'num_trades': len(self.trades),
            'final_equity': equity_curve[-1]
        }
    
    def plot_results(self, save_path: str = None):
        """結果をプロット"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 資産曲線
        ax1.plot(self.equity_curve, label='Equity')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Backtest Results')
        ax1.legend()
        ax1.grid(True)
        
        # ドローダウン
        equity_curve = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax * 100
        ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved backtest plot to {save_path}")
        else:
            plt.show()

def main():
    """メイン関数"""
    # ここに実際の価格と予測値を読み込むコードを追加
    logger.info("Running backtest...")
    
    # ダミーデータでテスト
    prices = np.random.randn(100).cumsum() + 100000
    predictions = prices + np.random.randn(100) * 1000
    
    backtester = Backtester(initial_capital=10000)
    backtester.run_backtest(prices, predictions)
    
    metrics = backtester.calculate_metrics()
    logger.info("Backtest metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")
    
    backtester.plot_results()

if __name__ == "__main__":
    main()