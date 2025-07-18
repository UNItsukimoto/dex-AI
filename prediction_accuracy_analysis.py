#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
予測精度の詳細分析 - 実際の市場動向との比較
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.integrated_backtest import IntegratedBacktester
from src.utils import get_logger

logger = get_logger(__name__)

class PredictionAnalyzer:
    """予測精度の詳細分析クラス"""
    
    def __init__(self):
        self.backtester = IntegratedBacktester(initial_capital=10000)
        
    async def load_and_predict(self, symbol='BTC', days_back=30):
        """データ読み込みと予測"""
        # モデルの読み込み
        if not self.backtester.load_model():
            logger.error("Failed to load model")
            return None
        
        # データの準備
        df_features, df_numeric, features_scaled = await self.backtester.prepare_backtest_data(
            symbol=symbol, 
            days_back=days_back
        )
        
        if df_features is None:
            return None
        
        # 予測の生成
        actual_prices, predicted_prices, timestamps = self.backtester.generate_predictions(
            features_scaled, df_features
        )
        
        return {
            'actual': actual_prices,
            'predicted': predicted_prices,
            'timestamps': timestamps,
            'df_features': df_features
        }
    
    def analyze_prediction_accuracy(self, data):
        """予測精度の詳細分析"""
        actual = data['actual']
        predicted = data['predicted']
        
        # 基本的な誤差指標
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        
        # 方向性の精度
        actual_returns = np.diff(actual)
        predicted_returns = predicted[1:] - actual[:-1]
        direction_accuracy = np.mean(np.sign(actual_returns) == np.sign(predicted_returns)) * 100
        
        # 相関分析
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        # ボラティリティの比較
        actual_volatility = np.std(np.diff(actual) / actual[:-1])
        predicted_volatility = np.std(np.diff(predicted) / predicted[:-1])
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'Direction Accuracy': direction_accuracy,
            'Correlation': correlation,
            'Actual Volatility': actual_volatility,
            'Predicted Volatility': predicted_volatility
        }
        
        return metrics
    
    def analyze_by_market_condition(self, data):
        """市場状況別の分析"""
        actual = data['actual']
        predicted = data['predicted']
        
        # 市場状況の分類
        returns = np.diff(actual) / actual[:-1]
        
        # トレンドの識別（移動平均を使用）
        ma_short = pd.Series(actual).rolling(5).mean()
        ma_long = pd.Series(actual).rolling(20).mean()
        
        # 市場状況の分類
        conditions = []
        for i in range(1, len(actual)):
            if i < 20:
                conditions.append('insufficient_data')
            elif ma_short.iloc[i] > ma_long.iloc[i] and returns[i-1] > 0.001:
                conditions.append('uptrend')
            elif ma_short.iloc[i] < ma_long.iloc[i] and returns[i-1] < -0.001:
                conditions.append('downtrend')
            else:
                conditions.append('sideways')
        
        # 各市場状況での精度
        condition_metrics = {}
        for condition in ['uptrend', 'downtrend', 'sideways']:
            indices = [i for i, c in enumerate(conditions) if c == condition]
            if len(indices) > 0:
                actual_subset = actual[indices]
                predicted_subset = predicted[indices]
                
                mape = np.mean(np.abs((actual_subset - predicted_subset) / actual_subset)) * 100
                direction_acc = self._calculate_direction_accuracy(actual_subset, predicted_subset)
                
                condition_metrics[condition] = {
                    'count': len(indices),
                    'mape': mape,
                    'direction_accuracy': direction_acc
                }
        
        return condition_metrics
    
    def _calculate_direction_accuracy(self, actual, predicted):
        """方向性の精度を計算"""
        if len(actual) < 2:
            return 0
        
        actual_changes = np.sign(np.diff(actual))
        predicted_changes = np.sign(predicted[1:] - actual[:-1])
        return np.mean(actual_changes == predicted_changes) * 100
    
    def analyze_time_horizon(self, data):
        """予測時間軸による分析"""
        df_features = data['df_features']
        actual = data['actual']
        predicted = data['predicted']
        
        # 異なる時間軸での精度
        horizons = {
            '1h': 1,
            '4h': 4,
            '12h': 12,
            '24h': 24
        }
        
        horizon_metrics = {}
        
        for name, hours in horizons.items():
            if hours < len(actual):
                # N時間後の実際の価格変化
                actual_future = actual[hours:]
                actual_current = actual[:-hours]
                actual_changes = (actual_future - actual_current) / actual_current
                
                # 予測された価格変化
                predicted_current = predicted[:-hours]
                predicted_changes = (predicted_current - actual_current) / actual_current
                
                # 相関と精度
                if len(actual_changes) > 0:
                    correlation = np.corrcoef(actual_changes, predicted_changes)[0, 1]
                    direction_acc = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
                    
                    horizon_metrics[name] = {
                        'correlation': correlation,
                        'direction_accuracy': direction_acc,
                        'sample_size': len(actual_changes)
                    }
        
        return horizon_metrics
    
    def plot_comprehensive_analysis(self, data, metrics, condition_metrics, horizon_metrics):
        """包括的な分析結果をプロット"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 実際vs予測価格
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(data['actual'], label='Actual', linewidth=2)
        ax1.plot(data['predicted'], label='Predicted', alpha=0.7)
        ax1.set_title('Price Comparison')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 散布図
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(data['actual'], data['predicted'], alpha=0.5)
        ax2.plot([data['actual'].min(), data['actual'].max()], 
                [data['actual'].min(), data['actual'].max()], 'r--', label='Perfect prediction')
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.set_title(f'Scatter Plot (R² = {metrics["R²"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 誤差分布
        ax3 = plt.subplot(3, 3, 3)
        errors = (data['predicted'] - data['actual']) / data['actual'] * 100
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Error Distribution (MAPE = {metrics["MAPE"]:.2f}%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 時系列誤差
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(np.abs(errors))
        ax4.axhline(y=metrics['MAPE'], color='r', linestyle='--', label=f'Average: {metrics["MAPE"]:.2f}%')
        ax4.set_ylabel('Absolute Error (%)')
        ax4.set_title('Prediction Error Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 市場状況別精度
        ax5 = plt.subplot(3, 3, 5)
        if condition_metrics:
            conditions = list(condition_metrics.keys())
            mapes = [condition_metrics[c]['mape'] for c in conditions]
            counts = [condition_metrics[c]['count'] for c in conditions]
            
            bars = ax5.bar(conditions, mapes)
            ax5.set_ylabel('MAPE (%)')
            ax5.set_title('Accuracy by Market Condition')
            
            # サンプル数を表示
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'n={count}', ha='center', va='bottom')
        
        # 6. 方向性精度（市場状況別）
        ax6 = plt.subplot(3, 3, 6)
        if condition_metrics:
            dir_accs = [condition_metrics[c]['direction_accuracy'] for c in conditions]
            ax6.bar(conditions, dir_accs)
            ax6.axhline(y=50, color='r', linestyle='--', label='Random guess')
            ax6.set_ylabel('Direction Accuracy (%)')
            ax6.set_title('Direction Prediction by Market Condition')
            ax6.legend()
        
        # 7. 時間軸別相関
        ax7 = plt.subplot(3, 3, 7)
        if horizon_metrics:
            horizons = list(horizon_metrics.keys())
            correlations = [horizon_metrics[h]['correlation'] for h in horizons]
            ax7.bar(horizons, correlations)
            ax7.set_ylabel('Correlation')
            ax7.set_title('Prediction Correlation by Time Horizon')
            ax7.set_ylim(-1, 1)
            ax7.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 8. リターンの比較
        ax8 = plt.subplot(3, 3, 8)
        actual_returns = np.diff(data['actual']) / data['actual'][:-1] * 100
        predicted_returns = np.diff(data['predicted']) / data['predicted'][:-1] * 100
        ax8.scatter(actual_returns, predicted_returns, alpha=0.5)
        ax8.set_xlabel('Actual Returns (%)')
        ax8.set_ylabel('Predicted Returns (%)')
        ax8.set_title('Return Prediction Accuracy')
        ax8.grid(True, alpha=0.3)
        
        # 9. 累積誤差
        ax9 = plt.subplot(3, 3, 9)
        cumulative_error = np.cumsum(np.abs(errors))
        ax9.plot(cumulative_error)
        ax9.set_ylabel('Cumulative Absolute Error (%)')
        ax9.set_title('Cumulative Prediction Error')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/comprehensive_prediction_analysis.png', dpi=300)
        logger.info("Saved comprehensive analysis to data/results/comprehensive_prediction_analysis.png")
        
    def print_analysis_summary(self, metrics, condition_metrics, horizon_metrics):
        """分析結果のサマリーを出力"""
        print("\n" + "="*60)
        print("PREDICTION ACCURACY ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n1. Overall Metrics:")
        print("-"*30)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")
        
        print("\n2. Market Condition Analysis:")
        print("-"*30)
        if condition_metrics:
            for condition, metrics in condition_metrics.items():
                print(f"\n{condition.upper()}:")
                for key, value in metrics.items():
                    print(f"  {key:18}: {value:.2f}" if isinstance(value, float) else f"  {key:18}: {value}")
        
        print("\n3. Time Horizon Analysis:")
        print("-"*30)
        if horizon_metrics:
            for horizon, metrics in horizon_metrics.items():
                print(f"\n{horizon}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key:18}: {value:.3f}")
                    else:
                        print(f"  {key:18}: {value}")
        
        print("\n" + "="*60)

async def main():
    """メイン実行関数"""
    analyzer = PredictionAnalyzer()
    
    # データの読み込みと予測
    logger.info("Loading data and generating predictions...")
    data = await analyzer.load_and_predict(symbol='BTC', days_back=30)
    
    if data is None:
        return
    
    # 予測精度の分析
    logger.info("Analyzing prediction accuracy...")
    metrics = analyzer.analyze_prediction_accuracy(data)
    
    # 市場状況別の分析
    logger.info("Analyzing by market conditions...")
    condition_metrics = analyzer.analyze_by_market_condition(data)
    
    # 時間軸別の分析
    logger.info("Analyzing by time horizons...")
    horizon_metrics = analyzer.analyze_time_horizon(data)
    
    # 結果の可視化
    logger.info("Creating visualizations...")
    analyzer.plot_comprehensive_analysis(data, metrics, condition_metrics, horizon_metrics)
    
    # サマリーの出力
    analyzer.print_analysis_summary(metrics, condition_metrics, horizon_metrics)
    
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    asyncio.run(main())