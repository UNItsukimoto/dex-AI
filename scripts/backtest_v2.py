#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改良版モデル（リターン予測）のバックテスト
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

class ReturnBasedBacktester:
    """リターンベースモデル用バックテスター"""
    
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
        
    def load_model(self):
        """訓練済みモデルを読み込む"""
        try:
            # メタデータの読み込み
            with open(f"{self.model_dir}/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {self.metadata['symbol']}, prediction type: {self.metadata['prediction_type']}")
            
            # スケーラーの読み込み
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # モデルの作成
            self.model = CryptoGAN(
                input_size=self.metadata['input_size'],
                sequence_length=self.metadata['sequence_length']
            )
            
            # 最新のチェックポイントを探す
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
            import traceback
            traceback.print_exc()
            return False
    
    async def prepare_backtest_data(self, symbol: str = 'BTC', days_back: int = 7):
        """バックテスト用のデータを準備"""
        # データローダーの初期化
        loader = HyperliquidDataLoader()
        feature_manager = FeatureManager()
        preprocessor = ImprovedDataPreprocessor(scaler_type='minmax')
        
        # データの取得
        df = await loader.download_historical_data(symbol, '1h', days_back)
        if df.empty:
            logger.error("No data downloaded")
            return None, None, None, None
        
        # 実際の価格を保存（バックテスト用）
        actual_prices = df['close'].values
        
        # 特徴量の作成
        df_features = feature_manager.create_all_features(df)
        
        # リターンベースの前処理
        df_features = preprocessor.prepare_return_based_features(df_features)
        
        # 実際のリターンを保存
        actual_returns = df_features['returns'].values
        
        # 数値列のみを選択
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
        df_numeric = df_features[feature_columns]
        
        # NaNと無限大の処理
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(0)
        
        # スケーリング
        features_scaled = self.scaler.transform(df_numeric.values)
        
        logger.info(f"Prepared data shape: {features_scaled.shape}")
        
        return df_features, features_scaled, actual_prices[1:], actual_returns  # 最初の価格はリターン計算で消える
    
    def generate_predictions(self, features_scaled, df_features):
        """モデルを使用してリターンを予測"""
        predicted_returns = []
        timestamps = []
        
        sequence_length = self.metadata['sequence_length']
        
        for i in range(sequence_length, len(features_scaled)):
            # シーケンスの作成
            sequence = features_scaled[i-sequence_length:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # 予測
            with torch.no_grad():
                # Generatorで生成
                noise = self.model.generate_noise(1)
                hidden = self.model.generator.init_hidden(1)
                output, _, _ = self.model.generator(noise, hidden)
                
                # 予測されたリターン（最初の要素と仮定）
                predicted_return = output.squeeze()[0].item()
            
            predicted_returns.append(predicted_return)
            timestamps.append(df_features.index[i])
        
        return np.array(predicted_returns), timestamps
    
    def run_backtest(self, actual_prices, actual_returns, predicted_returns, timestamps):
        """リターン予測を使用してバックテストを実行"""
        self.reset()
        self.actual_returns = actual_returns[self.metadata['sequence_length']:]
        self.predictions = predicted_returns
        self.timestamps = timestamps
        
        for i in range(len(predicted_returns)):
            current_price = actual_prices[self.metadata['sequence_length'] + i]
            predicted_return = predicted_returns[i]
            actual_return = self.actual_returns[i]
            
            # 現在の資産価値
            current_equity = self.capital + self.position * current_price
            self.equity_curve.append(current_equity)
            
            # 取引シグナル（予測リターンの閾値）
            threshold = 0.002  # 0.2%
            
            if predicted_return > threshold and self.position == 0:
                # 買いシグナル
                self.buy(current_price)
            elif predicted_return < -threshold and self.position > 0:
                # 売りシグナル
                self.sell(current_price)
        
        return self.calculate_metrics()
    
    def buy(self, price: float):
        """買い注文"""
        size = self.capital / price * (1 - self.fee_rate)
        cost = size * price * (1 + self.fee_rate)
        
        if cost <= self.capital:
            self.capital -= cost
            self.position += size
            self.trades.append({
                'type': 'buy',
                'price': price,
                'size': size,
                'timestamp': datetime.now()
            })
            logger.info(f"Buy: {size:.4f} @ ${price:.2f}")
    
    def sell(self, price: float):
        """売り注文"""
        if self.position > 0:
            revenue = self.position * price * (1 - self.fee_rate)
            self.capital += revenue
            self.trades.append({
                'type': 'sell',
                'price': price,
                'size': self.position,
                'timestamp': datetime.now()
            })
            logger.info(f"Sell: {self.position:.4f} @ ${price:.2f}")
            self.position = 0
    
    def calculate_metrics(self) -> dict:
        """パフォーマンスメトリクスを計算"""
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # 総リターン
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # シャープレシオ
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # 最大ドローダウン
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # 方向性精度
        if len(self.predictions) > 0 and len(self.actual_returns) > 0:
            direction_accuracy = np.mean(
                np.sign(self.predictions) == np.sign(self.actual_returns)
            ) * 100
        else:
            direction_accuracy = 0
        
        # 予測精度
        mae = np.mean(np.abs(self.predictions - self.actual_returns))
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'num_trades': len(self.trades),
            'final_equity': equity_curve[-1],
            'direction_accuracy': direction_accuracy,
            'mae_returns': mae
        }
    
    def plot_results(self, save_path: str = None):
        """バックテスト結果をプロット"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # 1. 資産曲線
        ax = axes[0]
        ax.plot(self.equity_curve, label='Portfolio Value')
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital')
        ax.set_ylabel('Equity ($)')
        ax.set_title('Portfolio Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 予測リターン vs 実際のリターン
        ax = axes[1]
        x = range(len(self.predictions))
        ax.plot(x, self.actual_returns, label='Actual Returns', alpha=0.7)
        ax.plot(x, self.predictions, label='Predicted Returns', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Returns')
        ax.set_title('Return Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 方向性の精度
        ax = axes[2]
        correct = np.sign(self.predictions) == np.sign(self.actual_returns)
        window = 50
        rolling_accuracy = pd.Series(correct).rolling(window).mean() * 100
        ax.plot(rolling_accuracy, label=f'{window}-period Rolling Accuracy')
        ax.axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Directional Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. ドローダウン
        ax = axes[3]
        equity_curve = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax * 100
        ax.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Time')
        ax.set_title('Drawdown Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved backtest plot to {save_path}")
        else:
            plt.show()

async def main():
    """メイン関数"""
    logger.info("Starting backtest for improved model...")
    
    # バックテスターの初期化
    backtester = ReturnBasedBacktester(initial_capital=10000)
    
    # モデルの読み込み
    if not backtester.load_model():
        logger.error("Failed to load model.")
        return
    
    # データの準備
    logger.info("Preparing backtest data...")
    df_features, features_scaled, actual_prices, actual_returns = await backtester.prepare_backtest_data(
        symbol='BTC', 
        days_back=14  # 2週間でテスト
    )
    
    if df_features is None:
        logger.error("Failed to prepare data")
        return
    
    # 予測の生成
    logger.info("Generating predictions...")
    predicted_returns, timestamps = backtester.generate_predictions(features_scaled, df_features)
    
    logger.info(f"Generated {len(predicted_returns)} predictions")
    
    # バックテストの実行
    logger.info("Running backtest...")
    metrics = backtester.run_backtest(actual_prices, actual_returns, predicted_returns, timestamps)
    
    # 結果の表示
    logger.info("=" * 50)
    logger.info("Backtest Results (Return-based Model):")
    logger.info("=" * 50)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.2f}")
    logger.info("=" * 50)
    
    # 結果のプロット
    backtester.plot_results(save_path='data/results/backtest_v2_results.png')

if __name__ == "__main__":
    asyncio.run(main())