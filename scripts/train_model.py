#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
モデル訓練スクリプト
"""

import sys
from pathlib import Path
import asyncio
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import HyperliquidDataLoader, DataPreprocessor
from src.features import FeatureManager
from src.models import CryptoGAN, GANTrainer
from src.utils import get_logger

logger = get_logger(__name__)

async def train_full_model(symbol: str = 'BTC', 
                          days_back: int = 30,
                          epochs: int = 100,
                          save_dir: str = 'data/models'):
    """完全なモデル訓練パイプライン"""
    
    logger.info(f"Starting full training pipeline for {symbol}")
    
    # 1. データの取得
    logger.info("Step 1: Loading data...")
    loader = HyperliquidDataLoader()
    df = await loader.download_historical_data(symbol, '1h', days_back)
    
    if df.empty:
        logger.error("No data loaded")
        return
    
    # 2. 特徴量エンジニアリング
    logger.info("Step 2: Feature engineering...")
    feature_manager = FeatureManager()
    df_features = feature_manager.create_all_features(df)
    
    # 3. データ前処理
    logger.info("Step 3: Data preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='minmax')
    
    # 数値列のみを選択
    numeric_columns = df_features.select_dtypes(include=[np.number]).columns
    df_numeric = df_features[numeric_columns]
    
    # NaNの処理
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # スケーリング
    df_scaled = preprocessor.scale_features(df_numeric, fit=True)
    
    # NumPy配列に変換
    features = df_scaled.values.astype(np.float32)
    
    logger.info(f"Features shape: {features.shape}")
    
    # 4. モデルの作成
    logger.info("Step 4: Creating model...")
    input_size = features.shape[1]
    sequence_length = 24
    
    gan = CryptoGAN(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=256,
        learning_rate_g=0.0002,
        learning_rate_d=0.0002
    )
    
    # 5. 訓練
    logger.info("Step 5: Training...")
    trainer = GANTrainer(gan, checkpoint_dir=f"{save_dir}/checkpoints")
    train_loader, val_loader = trainer.prepare_data(features, sequence_length)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        d_steps=1,
        g_steps=1,
        save_interval=10
    )
    
    # 6. 訓練履歴の保存
    logger.info("Step 6: Saving results...")
    
    # 履歴のプロット
    trainer.plot_history(save_path=f"{save_dir}/training_history.png")
    
    # メタデータの保存
    metadata = {
        'symbol': symbol,
        'days_back': days_back,
        'epochs': epochs,
        'input_size': input_size,
        'sequence_length': sequence_length,
        'feature_columns': numeric_columns.tolist(),
        'training_date': datetime.now().isoformat(),
        'final_d_loss': trainer.train_history['d_loss'][-1] if trainer.train_history['d_loss'] else None,
        'final_g_loss': trainer.train_history['g_loss'][-1] if trainer.train_history['g_loss'] else None
    }
    
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # スケーラーの保存
    import pickle
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    
    logger.info("Training completed!")
    
    return gan, preprocessor

def main():
    parser = argparse.ArgumentParser(description='Train Crypto GAN model')
    parser.add_argument('--symbol', type=str, default='BTC', help='Crypto symbol')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--save-dir', type=str, default='data/models', help='Save directory')
    
    args = parser.parse_args()
    
    asyncio.run(train_full_model(
        symbol=args.symbol,
        days_back=args.days,
        epochs=args.epochs,
        save_dir=args.save_dir
    ))

if __name__ == "__main__":
    main()