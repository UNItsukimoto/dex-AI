#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拡張特徴量の実装 - 元のGitHubプロジェクトに基づく
1. ARIMAによる予測を特徴量として追加
2. 相関資産の追加
3. フーリエ変換による周期性の抽出
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft, ifft
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import HyperliquidDataLoader
from src.utils import get_logger

logger = get_logger(__name__)

class EnhancedFeatureManager:
    """元のGitHubプロジェクトに基づく拡張特徴量マネージャー"""
    
    def __init__(self):
        self.loader = HyperliquidDataLoader()
        
    async def get_correlated_assets_features(self, symbol='BTC', interval='1h', days_back=30):
        """相関資産の特徴量を取得"""
        # 相関の高い資産を取得
        correlated_symbols = ['ETH', 'SOL', 'BNB']  # BTCと相関の高い仮想通貨
        
        all_data = {}
        
        # メインシンボルのデータ
        main_df = await self.loader.download_historical_data(symbol, interval, days_back)
        all_data[symbol] = main_df['close']
        
        # 相関資産のデータ
        for corr_symbol in correlated_symbols:
            try:
                df = await self.loader.download_historical_data(corr_symbol, interval, days_back)
                if not df.empty:
                    all_data[corr_symbol] = df['close']
                    logger.info(f"Loaded {corr_symbol} data: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {corr_symbol}: {e}")
        
        # DataFrameに結合
        corr_df = pd.DataFrame(all_data)
        
        # 相関係数を計算
        correlations = corr_df.corr()[symbol].drop(symbol)
        logger.info(f"Correlations with {symbol}:")
        for sym, corr in correlations.items():
            logger.info(f"  {sym}: {corr:.3f}")
        
        # 相関資産のリターンを特徴量として追加
        features = pd.DataFrame(index=corr_df.index)
        for col in corr_df.columns:
            features[f'{col}_returns'] = corr_df[col].pct_change()
            features[f'{col}_log_returns'] = np.log(corr_df[col] / corr_df[col].shift(1))
        
        return features
    
    def add_arima_predictions(self, df, column='close', order=(5, 1, 0)):
        """ARIMAモデルの予測を特徴量として追加"""
        logger.info("Adding ARIMA predictions as features...")
        
        # ARIMAモデルの予測を格納
        arima_predictions = []
        arima_residuals = []
        
        # 最小限のデータ数を確保
        min_train_size = 100
        
        for i in range(len(df)):
            if i < min_train_size:
                arima_predictions.append(np.nan)
                arima_residuals.append(np.nan)
            else:
                try:
                    # 過去のデータでモデルを訓練
                    train_data = df[column].iloc[:i]
                    model = ARIMA(train_data, order=order)
                    model_fit = model.fit()
                    
                    # 次の値を予測
                    prediction = model_fit.forecast(steps=1)[0]
                    arima_predictions.append(prediction)
                    
                    # 残差
                    if i < len(df) - 1:
                        actual_next = df[column].iloc[i]
                        residual = actual_next - prediction
                        arima_residuals.append(residual)
                    else:
                        arima_residuals.append(np.nan)
                        
                except Exception as e:
                    logger.warning(f"ARIMA failed at index {i}: {e}")
                    arima_predictions.append(np.nan)
                    arima_residuals.append(np.nan)
        
        # DataFrameに追加
        df['arima_prediction'] = arima_predictions
        df['arima_residual'] = arima_residuals
        df['arima_signal'] = df['arima_prediction'] > df[column]
        
        logger.info(f"Added ARIMA features. Non-null predictions: {df['arima_prediction'].notna().sum()}")
        
        return df
    
    def add_fourier_features(self, df, column='close', n_components=10):
        """フーリエ変換による周期性の特徴量"""
        logger.info("Adding Fourier transform features...")
        
        values = df[column].values
        
        # フーリエ変換
        fft_values = fft(values)
        
        # 主要な周波数成分を抽出
        fft_abs = np.abs(fft_values)
        frequencies = np.fft.fftfreq(len(values))
        
        # 上位N個の周波数成分を特徴として使用
        top_freq_idx = np.argsort(fft_abs)[-n_components:]
        
        # 各周波数成分の強度と位相を特徴量として追加
        for i, idx in enumerate(top_freq_idx):
            df[f'fourier_magnitude_{i}'] = fft_abs[idx] / len(values)
            df[f'fourier_phase_{i}'] = np.angle(fft_values[idx])
            df[f'fourier_freq_{i}'] = frequencies[idx]
        
        # トレンド成分の抽出（低周波成分の再構成）
        fft_filtered = np.zeros_like(fft_values)
        fft_filtered[:n_components] = fft_values[:n_components]
        fft_filtered[-n_components:] = fft_values[-n_components:]
        trend = np.real(ifft(fft_filtered))
        
        df['fourier_trend'] = trend
        df['fourier_detrended'] = values - trend
        
        logger.info(f"Added {n_components * 3 + 2} Fourier features")
        
        return df
    
    def add_market_regime_features(self, df, column='close', window=20):
        """市場レジームの特徴量（トレンド、ボラティリティレジーム）"""
        logger.info("Adding market regime features...")
        
        # リターンの計算
        returns = df[column].pct_change()
        
        # ボラティリティレジーム
        rolling_vol = returns.rolling(window).std()
        vol_regime = pd.qcut(rolling_vol, q=3, labels=['low_vol', 'mid_vol', 'high_vol'])
        df['volatility_regime'] = vol_regime
        
        # トレンドレジーム（移動平均のクロス）
        ma_short = df[column].rolling(window).mean()
        ma_long = df[column].rolling(window * 3).mean()
        
        df['trend_regime'] = np.where(ma_short > ma_long, 1, -1)
        df['trend_strength'] = (ma_short - ma_long) / ma_long
        
        # モメンタムレジーム
        momentum = df[column] / df[column].shift(window) - 1
        df['momentum_regime'] = np.where(momentum > 0.05, 1, 
                                        np.where(momentum < -0.05, -1, 0))
        
        logger.info("Added market regime features")
        
        return df

async def test_enhanced_features():
    """拡張特徴量のテスト"""
    manager = EnhancedFeatureManager()
    
    # 1. 相関資産の特徴量
    logger.info("Testing correlated assets features...")
    corr_features = await manager.get_correlated_assets_features('BTC', '1h', days_back=7)
    logger.info(f"Correlated features shape: {corr_features.shape}")
    logger.info(f"Features: {corr_features.columns.tolist()[:10]}...")
    
    # 2. 基本データの取得
    loader = HyperliquidDataLoader()
    df = await loader.download_historical_data('BTC', '1h', days_back=30)
    
    # 3. ARIMA特徴量
    logger.info("\nTesting ARIMA features...")
    df = manager.add_arima_predictions(df, 'close')
    
    # 4. フーリエ特徴量
    logger.info("\nTesting Fourier features...")
    df = manager.add_fourier_features(df, 'close')
    
    # 5. 市場レジーム特徴量
    logger.info("\nTesting market regime features...")
    df = manager.add_market_regime_features(df, 'close')
    
    # 結果の保存
    logger.info(f"\nFinal feature set: {df.shape}")
    logger.info(f"All features: {df.columns.tolist()}")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # 価格とARIMA予測
    ax = axes[0]
    ax.plot(df.index[-100:], df['close'][-100:], label='Actual', linewidth=2)
    ax.plot(df.index[-100:], df['arima_prediction'][-100:], label='ARIMA Prediction', alpha=0.7)
    ax.set_title('Price vs ARIMA Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # フーリエトレンド
    ax = axes[1]
    ax.plot(df.index[-100:], df['close'][-100:], label='Actual', alpha=0.7)
    ax.plot(df.index[-100:], df['fourier_trend'][-100:], label='Fourier Trend', linewidth=2)
    ax.set_title('Fourier Trend Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ボラティリティレジーム
    ax = axes[2]
    rolling_vol = df['close'].pct_change().rolling(20).std()
    ax.plot(df.index[-100:], rolling_vol[-100:] * 100, label='Rolling Volatility %')
    ax.set_title('Volatility Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # トレンド強度
    ax = axes[3]
    ax.plot(df.index[-100:], df['trend_strength'][-100:] * 100, label='Trend Strength %')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Trend Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/results/enhanced_features_test.png')
    logger.info("Saved visualization to data/results/enhanced_features_test.png")

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())