#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合バックテストスクリプト - GANモデルの予測を使用
既存のbacktest.pyを拡張した新しいバージョン
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

logger = get_logger(__name__)

class IntegratedBacktester:
    """既存のBacktesterクラスを拡張"""
    
    def __init__(self, initial_capital: float = 10000, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.model_dir = 'data/models'
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
        
    def load_model(self):
        """訓練済みモデルを読み込む（修正版）"""
        try:
            # メタデータの読み込み
            with open(f"{self.model_dir}/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {self.metadata['symbol']}, epochs: {self.metadata['epochs']}")
            
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
                # ファイル名パターンを修正
                checkpoints = list(checkpoint_dir.glob("gan_epoch_*_generator.pth"))
                if checkpoints:
                    # エポック番号でソートして最新を取得
                    def get_epoch_number(path):
                        # gan_epoch_100_generator.pth から 100 を抽出
                        name = path.stem  # gan_epoch_100_generator
                        parts = name.split('_')
                        return int(parts[2])  # epoch番号
                    
                    latest_checkpoint = max(checkpoints, key=get_epoch_number)
                    
                    # チェックポイントの読み込み（完全な形式に対応）
                    checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
                    
                    # チェックポイントの形式を確認
                    if isinstance(checkpoint, dict):
                        # 完全なチェックポイント形式の場合
                        if 'model_state_dict' in checkpoint:
                            self.model.generator.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                            
                            # メトリクスがあれば表示
                            if 'metrics' in checkpoint:
                                metrics = checkpoint['metrics']
                                d_loss = metrics.get('d_loss', 'N/A')
                                g_loss = metrics.get('g_loss', 'N/A')
                                
                                # 数値の場合のみフォーマット
                                d_loss_str = f"{d_loss:.4f}" if isinstance(d_loss, (int, float)) else str(d_loss)
                                g_loss_str = f"{g_loss:.4f}" if isinstance(g_loss, (int, float)) else str(g_loss)
                                
                                logger.info(f"Training metrics - D Loss: {d_loss_str}, G Loss: {g_loss_str}")
                        else:
                            # 古い形式（state_dictのみ）の場合
                            self.model.generator.load_state_dict(checkpoint)
                    else:
                        # state_dictが直接保存されている場合
                        self.model.generator.load_state_dict(checkpoint)
                    
                    self.model.generator.eval()
                    logger.info(f"Loaded checkpoint: {latest_checkpoint}")
                else:
                    logger.warning("No checkpoint found")
                    return False
            else:
                logger.error("Checkpoint directory not found")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def prepare_backtest_data(self, symbol: str = 'BTC', days_back: int = 7):
        """バックテスト用のデータを準備（訓練時と同じ処理を再現）"""
        # データローダーの初期化
        loader = HyperliquidDataLoader()
        feature_manager = FeatureManager()
        
        # データの取得
        df = await loader.download_historical_data(symbol, '1h', days_back)
        if df.empty:
            logger.error("No data downloaded")
            return None, None, None
        
        # 特徴量の作成
        df_features = feature_manager.create_all_features(df)
        
        # 数値列のみを選択（訓練時と同じ処理）
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_features[numeric_columns]
        
        # 無限大やNaNの処理
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.ffill().bfill().fillna(0)
        
        # float32に変換
        df_numeric = df_numeric.astype(np.float32)
        
        logger.info(f"Current features: {df_numeric.shape[1]} columns")
        logger.info(f"Model expects: {self.metadata['input_size']} columns")
        
        # 特徴量の数を調整
        if df_numeric.shape[1] > self.metadata['input_size']:
            # 多すぎる場合は最初のN個を使用
            logger.warning(f"Too many features ({df_numeric.shape[1]}), truncating to {self.metadata['input_size']}")
            df_numeric = df_numeric.iloc[:, :self.metadata['input_size']]
        elif df_numeric.shape[1] < self.metadata['input_size']:
            # 少ない場合はゼロパディング
            logger.warning(f"Too few features ({df_numeric.shape[1]}), padding to {self.metadata['input_size']}")
            padding_cols = self.metadata['input_size'] - df_numeric.shape[1]
            for i in range(padding_cols):
                df_numeric[f'padding_{i}'] = 0
        
        # スケーリング（スケーラーを使わずに簡易的なスケーリング）
        if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != df_numeric.shape[1]:
            logger.warning(f"Scaler expects {self.scaler.n_features_in_} features but got {df_numeric.shape[1]}")
            logger.warning("Using simple min-max scaling instead")
            
            # 簡易的なMin-Maxスケーリング
            features_scaled = df_numeric.values
            for i in range(features_scaled.shape[1]):
                col_min = features_scaled[:, i].min()
                col_max = features_scaled[:, i].max()
                if col_max > col_min:
                    features_scaled[:, i] = (features_scaled[:, i] - col_min) / (col_max - col_min)
        else:
            # 通常のスケーリング
            features_scaled = self.scaler.transform(df_numeric.values)
        
        logger.info(f"Final scaled data shape: {features_scaled.shape}")
        
        return df_features, df_numeric, features_scaled
    
    def generate_predictions(self, features_scaled, df_features):
        """モデルを使用して予測を生成"""
        predictions = []
        actual_prices = []
        timestamps = []
        
        sequence_length = self.metadata['sequence_length']
        
        for i in range(sequence_length, len(features_scaled)):
            # シーケンスの作成
            sequence = features_scaled[i-sequence_length:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # 予測
            with torch.no_grad():
                pred = self.model.predict(sequence_tensor, num_predictions=1)
            
            # 予測結果の処理
            # モデルは全特徴量を予測するので、価格関連の情報を抽出
            pred_values = pred.numpy()[0]
            
            # 現在の価格
            current_price = df_features['close'].iloc[i]
            
            # 予測価格の推定（複数の方法を試す）
            predicted_price = current_price  # デフォルト
            
            # 方法1: close列がある場合
            if 'close' in df_features.columns:
                close_idx = 3  # 通常、open, high, low, closeの順
                if close_idx < len(pred_values):
                    # スケールを戻す
                    pred_rescaled = pred_values[close_idx]
                    # 価格の範囲を推定
                    price_min = df_features['close'].min()
                    price_max = df_features['close'].max()
                    predicted_price = price_min + pred_rescaled * (price_max - price_min)
            
            # 方法2: リターンから計算
            if 'returns' in df_features.columns and i > 0:
                returns_idx = 6  # 通常の位置
                if returns_idx < len(pred_values):
                    predicted_return = pred_values[returns_idx]
                    # リターンは通常-1から1の範囲
                    predicted_return = np.clip(predicted_return, -0.1, 0.1)  # 10%以内に制限
                    predicted_price = current_price * (1 + predicted_return)
            
            predictions.append(predicted_price)
            actual_prices.append(current_price)
            timestamps.append(df_features.index[i])
        
        return np.array(actual_prices), np.array(predictions), timestamps
    
    def calculate_signals(self, current_price: float, predicted_price: float, 
                         threshold: float = 0.01):
        """取引シグナルを計算"""
        predicted_change = (predicted_price - current_price) / current_price
        
        if predicted_change > threshold and self.position == 0:
            return 'buy'
        elif predicted_change < -threshold and self.position > 0:
            return 'sell'
        else:
            return 'hold'
    
    def run_backtest_with_predictions(self, actual_prices, predicted_prices, timestamps):
        """予測を使用してバックテストを実行"""
        self.reset()
        
        for i in range(len(predicted_prices)):
            current_price = actual_prices[i]
            predicted_price = predicted_prices[i]
            
            # 現在の資産価値
            current_equity = self.capital + self.position * current_price
            self.equity_curve.append(current_equity)
            
            # シグナルの計算
            signal = self.calculate_signals(current_price, predicted_price)
            
            # 取引の実行
            if signal == 'buy':
                self.buy(current_price)
            elif signal == 'sell':
                self.sell(current_price)
        
        # メトリクスの計算
        metrics = self.calculate_metrics()
        
        # 予測精度の追加
        mae = np.mean(np.abs(predicted_prices - actual_prices))
        mape = np.mean(np.abs((predicted_prices - actual_prices) / actual_prices)) * 100
        
        metrics['mae'] = mae
        metrics['mape'] = mape
        
        return metrics
    
    def buy(self, price: float, size: float = None):
        """買い注文（既存のメソッドを使用）"""
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
        """売り注文（既存のメソッドを使用）"""
        if size is None:
            size = self.position
        
        if size <= self.position and size > 0:
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
    
    def calculate_metrics(self) -> dict:
        """パフォーマンスメトリクスを計算（既存のメソッドを拡張）"""
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
        profitable_trades = 0
        total_trades_with_pnl = 0
        
        for i in range(len(self.trades)):
            if self.trades[i]['type'] == 'sell' and i > 0:
                # 直前の買い注文を探す
                for j in range(i-1, -1, -1):
                    if self.trades[j]['type'] == 'buy':
                        buy_price = self.trades[j]['price']
                        sell_price = self.trades[i]['price']
                        if sell_price > buy_price:
                            profitable_trades += 1
                        total_trades_with_pnl += 1
                        break
        
        win_rate = profitable_trades / total_trades_with_pnl if total_trades_with_pnl > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'num_trades': len(self.trades),
            'final_equity': equity_curve[-1]
        }
    
    def plot_results(self, actual_prices, predicted_prices, save_path: str = None):
        """結果をプロット（予測vs実際の価格も含む）"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 価格予測の比較
        ax1 = axes[0]
        ax1.plot(actual_prices, label='Actual Price', color='blue')
        ax1.plot(predicted_prices, label='Predicted Price', color='red', alpha=0.7)
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Prediction vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 資産曲線
        ax2 = axes[1]
        ax2.plot(self.equity_curve, label='Portfolio Value')
        ax2.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital')
        ax2.set_ylabel('Equity ($)')
        ax2.set_title('Portfolio Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ドローダウン
        ax3 = axes[2]
        equity_curve = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax * 100
        ax3.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Time')
        ax3.set_title('Drawdown Analysis')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # ディレクトリが存在しない場合は作成
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path)
            logger.info(f"Saved backtest plot to {save_path}")
        else:
            plt.show()

async def main():
    """メイン関数"""
    logger.info("Starting integrated backtest...")
    
    # バックテスターの初期化
    backtester = IntegratedBacktester(initial_capital=10000)
    
    # モデルの読み込み
    if not backtester.load_model():
        logger.error("Failed to load model. Please train the model first.")
        return
    
    # データの準備
    logger.info("Preparing backtest data...")
    df_features, df_numeric, features_scaled = await backtester.prepare_backtest_data(
        symbol='BTC', 
        days_back=7
    )
    
    if df_features is None:
        logger.error("Failed to prepare data")
        return
    
    # 予測の生成
    logger.info("Generating predictions...")
    actual_prices, predicted_prices, timestamps = backtester.generate_predictions(
        features_scaled, df_features
    )
    
    logger.info(f"Generated {len(predicted_prices)} predictions")
    
    # バックテストの実行
    logger.info("Running backtest...")
    metrics = backtester.run_backtest_with_predictions(
        actual_prices, predicted_prices, timestamps
    )
    
    # 結果の表示
    logger.info("=" * 50)
    logger.info("Backtest Results:")
    logger.info("=" * 50)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.2f}")
    logger.info("=" * 50)
    
    # 結果のプロット
    backtester.plot_results(
        actual_prices, 
        predicted_prices,
        save_path='data/results/integrated_backtest.png'
    )

if __name__ == "__main__":
    asyncio.run(main())