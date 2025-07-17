#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量エンジニアリングのテストスクリプト
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import HyperliquidDataLoader
from src.features import FeatureManager
from src.utils import get_logger

logger = get_logger(__name__)

async def test_features():
    """特徴量エンジニアリングのテスト"""
    # データの取得
    loader = HyperliquidDataLoader()
    df = await loader.download_historical_data('BTC', '1h', days_back=30)
    
    if df.empty:
        logger.error("No data loaded")
        return
        
    logger.info(f"Loaded data shape: {df.shape}")
    
    # 特徴量の作成
    feature_manager = FeatureManager()
    df_features = feature_manager.create_all_features(df)
    
    logger.info(f"Features created. Shape: {df_features.shape}")
    logger.info(f"Total features: {len(df_features.columns)}")
    
    # 特徴量の統計
    logger.info("\nFeature statistics:")
    print(df_features.describe())
    
    # 重要な特徴量の選択
    important_features = feature_manager.select_important_features(
        df_features, 
        target_col='close',
        n_features=20
    )
    
    logger.info(f"\nTop 20 important features:")
    for i, feature in enumerate(important_features, 1):
        correlation = df_features[feature].corr(df_features['close'])
        logger.info(f"{i}. {feature}: {correlation:.3f}")
    
    # 特徴量の可視化（サンプル）
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # 価格とテクニカル指標
    ax = axes[0, 0]
    ax.plot(df_features.index[-100:], df_features['close'][-100:], label='Close')
    ax.plot(df_features.index[-100:], df_features['sma_21'][-100:], label='SMA 21')
    ax.plot(df_features.index[-100:], df_features['bb_upper'][-100:], label='BB Upper', alpha=0.5)
    ax.plot(df_features.index[-100:], df_features['bb_lower'][-100:], label='BB Lower', alpha=0.5)
    ax.set_title('Price and Bollinger Bands')
    ax.legend()
    
    # RSI
    ax = axes[0, 1]
    ax.plot(df_features.index[-100:], df_features['rsi_14'][-100:])
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax.set_title('RSI (14)')
    ax.set_ylim(0, 100)
    
    # ボリューム
    ax = axes[1, 0]
    ax.bar(df_features.index[-100:], df_features['volume'][-100:])
    ax.set_title('Volume')
    
    # マーケットマイクロストラクチャー
    ax = axes[1, 1]
    ax.plot(df_features.index[-100:], df_features['amihud_illiquidity_ma'][-100:])
    ax.set_title('Amihud Illiquidity (MA)')
    
    # トレンド強度
    ax = axes[2, 0]
    ax.plot(df_features.index[-100:], df_features['trend_strength'][-100:])
    ax.set_title('Trend Strength (R²)')
    ax.set_ylim(0, 1)
    
    # Hurstエクスポネント
    ax = axes[2, 1]
    ax.plot(df_features.index[-100:], df_features['hurst_exponent'][-100:])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Hurst Exponent')
    
    plt.tight_layout()
    plt.savefig('features_visualization.png')
    logger.info("Saved visualization to features_visualization.png")
    
    # CSVに保存
    df_features.to_csv('data/processed/btc_features.csv')
    logger.info("Saved features to data/processed/btc_features.csv")
    
    logger.info("\n✅ Feature engineering test completed!")

def main():
    """メイン関数"""
    print("=" * 50)
    print("Feature Engineering Test")
    print("=" * 50)
    
    asyncio.run(test_features())

if __name__ == "__main__":
    main()