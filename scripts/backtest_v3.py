#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改善されたバックテストv3 - より現実的な予測と取引戦略
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger
from src.data import HyperliquidDataLoader
from src.features import FeatureManager
from src.models import CryptoGAN
from scripts.improved_training_returns import ImprovedDataPreprocessor

logger = get_logger(__name__)

class ImprovedBacktesterV3:
    """改善されたバックテスター - 適切なスケーリングと戦略"""
    
    def __init__(self, initial_capital: float = 10000, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.model_dir = 'data/models_v2'
        self.model = None
        self.scaler = None
        self.metadata = None
        self.reset()
        
    def reset(self):
        """状態をリセット"""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.predictions = []
        self.actual_returns = []
        self.timestamps = []
        self.positions_over_time = []
        
    def load_model(self):
        """訓練済みモデルを読み込む"""
        try:
            with open(f"{self.model_dir}/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {self.metadata['symbol']}")
            
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.model = CryptoGAN(
                input_size=self.metadata['input_size'],
                sequence_length=self.metadata['sequence_length']
            )
            
            checkpoint_dir = Path(f"{self.model_dir}/checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("gan_epoch_*_generator.pth"))
                if checkpoints:
                    def get_epoch_number(path):
                        name = path.stem
                        parts = name.split('_')
                        return int(parts[2])
                    
                    latest_checkpoint = max(checkpoints, key=get_epoch_number)
                    checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.generator.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.generator.load_state_dict(checkpoint)
                    
                    self.model.generator.eval()
                    logger.info(f"Loaded checkpoint: {latest_checkpoint}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def prepare_backtest_data(self, symbol: str = 'BTC', days_back: int = 7):
        """バックテスト用のデータを準備"""
        loader = HyperliquidDataLoader()
        feature_manager = FeatureManager()
        preprocessor = ImprovedDataPreprocessor(scaler_type='minmax')
        
        df = await loader.download_historical_data(symbol, '1h', days_back)
        if df.empty:
            logger.error("No data downloaded")
            return None, None, None, None
        
        actual_prices = df['close'].values
        
        df_features = feature_manager.create_all_features(df)
        df_features = preprocessor.prepare_return_based_features(df_features)
        
        actual_returns = df_features['returns'].values
        
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
        df_numeric = df_features[feature_columns]
        
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(0)
        
        features_scaled = self.scaler.transform(df_numeric.values)
        
        logger.info(f"Prepared data shape: {features_scaled.shape}")
        logger.info(f"Actual returns std: {np.std(actual_returns):.6f}")
        
        return df_features, features_scaled, actual_prices[1:], actual_returns
    
    def generate_predictions_improved(self, features_scaled, df_features):
        """改善された予測生成 - より現実的なアプローチ"""
        predicted_returns = []
        raw_outputs = []
        timestamps = []
        
        sequence_length = self.metadata['sequence_length']
        
        # 実際のリターンの統計
        actual_returns = df_features['returns'].values
        actual_std = np.std(actual_returns[~np.isnan(actual_returns)])
        actual_mean = np.mean(actual_returns[~np.isnan(actual_returns)])
        
        logger.info(f"Actual returns - mean: {actual_mean:.6f}, std: {actual_std:.6f}")
        
        # モデルの出力を実際のシーケンスベースで生成
        for i in range(sequence_length, len(features_scaled)):
            # 実際のシーケンスを使用
            sequence = features_scaled[i-sequence_length:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                # シーケンスベースの予測を試みる
                prediction = self.model.predict(sequence_tensor, num_predictions=1)
                
                # 最初の要素をリターン予測として使用
                if prediction.shape[1] > 0:
                    raw_output = prediction[0, 0].item()
                else:
                    # フォールバック：ノイズから生成
                    noise = self.model.generate_noise(1)
                    hidden = self.model.generator.init_hidden(1)
                    output, _, _ = self.model.generator(noise, hidden)
                    raw_output = output.squeeze()[0].item()
            
            raw_outputs.append(raw_output)
            
            # スケーリングとノイズの追加でより現実的に
            # 1. 基本的なスケーリング（モデル出力は既に適切な範囲の可能性）
            scaled_output = raw_output
            
            # 2. 小さなノイズを追加して多様性を増やす
            noise_level = actual_std * 0.1  # 実際の標準偏差の10%のノイズ
            scaled_output += np.random.normal(0, noise_level)
            
            # 3. 極端な値をクリップ
            scaled_output = np.clip(scaled_output, -3*actual_std, 3*actual_std)
            
            predicted_returns.append(scaled_output)
            timestamps.append(df_features.index[i])
        
        # 統計情報の出力
        pred_std = np.std(predicted_returns)
        pred_mean = np.mean(predicted_returns)
        raw_std = np.std(raw_outputs)
        
        logger.info(f"Raw model outputs - mean: {np.mean(raw_outputs):.6f}, std: {raw_std:.6f}")
        logger.info(f"Scaled predictions - mean: {pred_mean:.6f}, std: {pred_std:.6f}")
        
        return np.array(predicted_returns), timestamps
    
    def run_backtest_with_improved_strategy(self, actual_prices, actual_returns, predicted_returns, timestamps):
        """改善された取引戦略でバックテストを実行"""
        self.reset()
        self.actual_returns = actual_returns[self.metadata['sequence_length']:]
        self.predictions = predicted_returns
        self.timestamps = timestamps
        
        # 動的閾値の計算
        base_threshold = 0.001  # 0.1%
        
        for i in range(len(predicted_returns)):
            current_price = actual_prices[self.metadata['sequence_length'] + i]
            predicted_return = predicted_returns[i]
            actual_return = self.actual_returns[i]
            
            # 現在の資産価値
            current_equity = self.capital + self.position * current_price
            self.equity_curve.append(current_equity)
            self.positions_over_time.append(self.position)
            
            # ローリングボラティリティに基づく動的閾値
            if i > 20:
                recent_vol = np.std(self.actual_returns[max(0, i-20):i])
                threshold = max(base_threshold, recent_vol * 0.5)
            else:
                threshold = base_threshold
            
            # 信頼度に基づくポジションサイジング
            confidence = min(abs(predicted_return) / threshold, 1.0)
            
            # 取引ロジック
            if predicted_return > threshold:
                if self.position == 0:
                    # 新規買い（信頼度に基づくサイズ）
                    self.buy(current_price, size_ratio=confidence * 0.8)
                elif self.position < self.capital * 0.8 / current_price:
                    # 追加買い（ポジションが小さい場合）
                    self.buy(current_price, size_ratio=confidence * 0.2)
                    
            elif predicted_return < -threshold:
                if self.position > 0:
                    # 売り（信頼度に基づく部分決済）
                    if confidence > 0.7:
                        self.sell(current_price, size_ratio=1.0)  # 全売却
                    else:
                        self.sell(current_price, size_ratio=0.5)  # 半分売却
        
        return self.calculate_metrics()
    
    def buy(self, price: float, size_ratio: float = 1.0):
        """買い注文（サイズ比率付き）"""
        available_capital = self.capital * size_ratio
        size = available_capital / price * (1 - self.fee_rate)
        cost = size * price * (1 + self.fee_rate)
        
        if cost <= self.capital and size > 0:
            self.capital -= cost
            self.position += size
            self.trades.append({
                'type': 'buy',
                'price': price,
                'size': size,
                'timestamp': self.timestamps[len(self.equity_curve) - 2] if len(self.equity_curve) > 1 else datetime.now()
            })
            logger.info(f"Buy: {size:.4f} @ ${price:.2f} (ratio: {size_ratio:.2f})")
    
    def sell(self, price: float, size_ratio: float = 1.0):
        """売り注文（サイズ比率付き）"""
        size = self.position * size_ratio
        if size > 0:
            revenue = size * price * (1 - self.fee_rate)
            self.capital += revenue
            self.position -= size
            self.trades.append({
                'type': 'sell',
                'price': price,
                'size': size,
                'timestamp': self.timestamps[len(self.equity_curve) - 2] if len(self.equity_curve) > 1 else datetime.now()
            })
            logger.info(f"Sell: {size:.4f} @ ${price:.2f} (ratio: {size_ratio:.2f})")
    
    def calculate_metrics(self) -> dict:
        """パフォーマンスメトリクスを計算"""
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        if len(self.predictions) > 0 and len(self.actual_returns) > 0:
            direction_accuracy = np.mean(
                np.sign(self.predictions) == np.sign(self.actual_returns)
            ) * 100
            
            # 取引時の精度も計算
            trade_accuracy = self.calculate_trade_accuracy()
        else:
            direction_accuracy = 0
            trade_accuracy = 0
        
        mae = np.mean(np.abs(self.predictions - self.actual_returns))
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'num_trades': len(self.trades),
            'final_equity': equity_curve[-1],
            'direction_accuracy': direction_accuracy,
            'trade_accuracy': trade_accuracy,
            'mae_returns': mae
        }
    
    def calculate_trade_accuracy(self):
        """実際の取引の精度を計算"""
        correct_trades = 0
        total_evaluated = 0
        
        for i, trade in enumerate(self.trades):
            if trade['type'] == 'buy' and i < len(self.trades) - 1:
                # 次の売り取引を探す
                for j in range(i + 1, len(self.trades)):
                    if self.trades[j]['type'] == 'sell':
                        if self.trades[j]['price'] > trade['price']:
                            correct_trades += 1
                        total_evaluated += 1
                        break
        
        return (correct_trades / total_evaluated * 100) if total_evaluated > 0 else 0
    
    def plot_enhanced_results(self, save_path: str = None):
        """拡張された結果のプロット"""
        fig, axes = plt.subplots(5, 1, figsize=(14, 16))
        
        # 1. 資産曲線と取引ポイント
        ax = axes[0]
        ax.plot(self.equity_curve, label='Portfolio Value', linewidth=2)
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital', alpha=0.7)
        
        # 取引マーカー
        buy_indices = []
        sell_indices = []
        for i, trade in enumerate(self.trades):
            # タイムスタンプからインデックスを見つける
            try:
                idx = self.timestamps.index(trade['timestamp']) + self.metadata['sequence_length']
                if trade['type'] == 'buy':
                    buy_indices.append(idx)
                else:
                    sell_indices.append(idx)
            except:
                pass
        
        if buy_indices:
            ax.scatter(buy_indices, [self.equity_curve[i] for i in buy_indices], 
                      color='g', marker='^', s=100, label='Buy', zorder=5)
        if sell_indices:
            ax.scatter(sell_indices, [self.equity_curve[i] for i in sell_indices], 
                      color='r', marker='v', s=100, label='Sell', zorder=5)
        
        ax.set_ylabel('Equity ($)')
        ax.set_title('Portfolio Performance with Trading Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 予測リターン vs 実際のリターン（改善版）
        ax = axes[1]
        x = range(len(self.predictions))
        ax.plot(x, self.actual_returns * 100, label='Actual Returns (%)', alpha=0.7, linewidth=1)
        ax.plot(x, self.predictions * 100, label='Predicted Returns (%)', alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Returns (%)')
        ax.set_title('Return Predictions vs Actual (Percentage)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. ポジションサイズの推移
        ax = axes[2]
        ax.plot(self.positions_over_time, linewidth=2, color='purple')
        ax.fill_between(range(len(self.positions_over_time)), 0, self.positions_over_time, 
                       alpha=0.3, color='purple')
        ax.set_ylabel('Position Size')
        ax.set_title('Position Size Over Time')
        ax.grid(True, alpha=0.3)
        
        # 4. ローリング精度
        ax = axes[3]
        correct = np.sign(self.predictions) == np.sign(self.actual_returns)
        window = 50
        rolling_accuracy = pd.Series(correct).rolling(window).mean() * 100
        ax.plot(rolling_accuracy, label=f'{window}-period Rolling Accuracy', linewidth=2)
        ax.axhline(y=50, color='r', linestyle='--', label='Random (50%)', alpha=0.7)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Directional Prediction Accuracy')
        ax.set_ylim(30, 70)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. ドローダウン
        ax = axes[4]
        equity_curve = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax * 100
        ax.fill_between(range(len(drawdown)), 0, drawdown, 
                       where=(drawdown < 0), color='red', alpha=0.3)
        ax.plot(drawdown, color='red', linewidth=1)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Time')
        ax.set_title('Drawdown Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved enhanced backtest plot to {save_path}")
        else:
            plt.show()

async def main():
    """メイン関数"""
    logger.info("Starting improved backtest v3...")
    
    backtester = ImprovedBacktesterV3(initial_capital=10000)
    
    if not backtester.load_model():
        logger.error("Failed to load model.")
        return
    
    logger.info("Preparing backtest data...")
    df_features, features_scaled, actual_prices, actual_returns = await backtester.prepare_backtest_data(
        symbol='BTC', 
        days_back=14
    )
    
    if df_features is None:
        logger.error("Failed to prepare data")
        return
    
    logger.info("Generating improved predictions...")
    predicted_returns, timestamps = backtester.generate_predictions_improved(features_scaled, df_features)
    
    logger.info(f"Generated {len(predicted_returns)} predictions")
    
    logger.info("Running backtest with improved strategy...")
    metrics = backtester.run_backtest_with_improved_strategy(
        actual_prices, actual_returns, predicted_returns, timestamps
    )
    
    logger.info("=" * 60)
    logger.info("IMPROVED BACKTEST RESULTS V3:")
    logger.info("=" * 60)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.2f}")
    logger.info("=" * 60)
    
    backtester.plot_enhanced_results(save_path='data/results/backtest_v3_enhanced.png')

if __name__ == "__main__":
    asyncio.run(main())