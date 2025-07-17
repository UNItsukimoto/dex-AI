#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
データローダーのテストスクリプト
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import HyperliquidDataLoader, DataPreprocessor
from src.utils import get_logger

logger = get_logger(__name__)

async def test_data_loader():
    """データローダーのテスト"""
    loader = HyperliquidDataLoader()
    
    try:
        # 1. 履歴データのダウンロード
        logger.info("Testing historical data download...")
        df = await loader.download_historical_data('BTC', '1h', days_back=7)
        
        if not df.empty:
            logger.info(f"Downloaded {len(df)} rows of data")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"\nFirst few rows:")
            print(df.head())
            
        # 2. マーケットデプスの取得
        logger.info("\nTesting market depth snapshot...")
        depth_data = await loader.get_market_depth_snapshot('BTC')
        
        if depth_data:
            features = depth_data['features']
            logger.info("Market depth features:")
            for key, value in features.items():
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # 3. 相関アセットの取得
        logger.info("\nTesting correlated assets...")
        corr_df = await loader.get_correlated_assets('BTC', '1h', days_back=7)
        
        if not corr_df.empty:
            logger.info(f"Correlated assets: {corr_df.columns.tolist()}")
            logger.info(f"Correlation matrix:")
            print(corr_df.corr())
            
        # 4. データ前処理のテスト
        logger.info("\nTesting data preprocessing...")
        preprocessor = DataPreprocessor()
        
        # 基本的な特徴量を追加
        df_with_features = preprocessor.add_basic_features(df)
        logger.info(f"Features after preprocessing: {df_with_features.columns.tolist()}")
        
        # スケーリング
        scaled_df = preprocessor.scale_features(df_with_features)
        logger.info(f"Shape after scaling: {scaled_df.shape}")
        
        # シーケンス作成
        sequences, targets = preprocessor.create_sequences(scaled_df, sequence_length=24)
        logger.info(f"Sequences shape: {sequences.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        
        logger.info("\n✅ All data loader tests passed!")
        
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン関数"""
    print("=" * 50)
    print("Hyperliquid Data Loader Test")
    print("=" * 50)
    
    asyncio.run(test_data_loader())

if __name__ == "__main__":
    main()