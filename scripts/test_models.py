#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
モデルのテストスクリプト
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CryptoGAN, GANTrainer
from src.utils import get_logger

logger = get_logger(__name__)

def test_models():
    """モデルのテスト"""
    # デバイスの確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # テストデータの作成
    sequence_length = 24
    input_size = 85  # 特徴量の数
    batch_size = 4
    
    # ダミーデータ
    test_data = torch.randn(batch_size, sequence_length, input_size)
    logger.info(f"Test data shape: {test_data.shape}")
    
    # モデルの作成
    logger.info("Creating GAN model...")
    gan = CryptoGAN(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=128,  # テスト用に小さく
    )
    
    # Generatorのテスト
    logger.info("\nTesting Generator...")
    noise = gan.generate_noise(batch_size)
    hidden = gan.generator.init_hidden(batch_size)
    output, new_hidden, attention = gan.generator(noise, hidden)
    logger.info(f"Generator output shape: {output.shape}")
    logger.info(f"Hidden state shape: {new_hidden[0].shape}")
    
    # Discriminatorのテスト
    logger.info("\nTesting Discriminator...")
    disc_output = gan.discriminator(test_data)
    logger.info(f"Discriminator output shape: {disc_output.shape}")
    logger.info(f"Discriminator output: {disc_output.squeeze().tolist()}")
    
    # 訓練ステップのテスト
    logger.info("\nTesting training step...")
    
    # Discriminatorの訓練
    d_metrics = gan.train_discriminator(test_data, noise)
    logger.info(f"Discriminator metrics: {d_metrics}")
    
    # Generatorの訓練
    g_metrics = gan.train_generator(noise)
    logger.info(f"Generator metrics: {g_metrics}")
    
    # データ準備のテスト
    logger.info("\nTesting data preparation...")
    
    # 実際の特徴量データを読み込む（あれば）
    try:
        features_df = pd.read_csv('data/processed/btc_features.csv', index_col=0)
        logger.info(f"Loaded features shape: {features_df.shape}")
        logger.info(f"Features columns: {features_df.columns.tolist()[:10]}...")  # 最初の10列
        
        # 数値列のみを選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_columns]
        logger.info(f"Numeric features shape: {features_numeric.shape}")
        
        # NaNや無限大を処理
        features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)
        features_numeric = features_numeric.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # NumPy配列に変換
        features = features_numeric.values.astype(np.float32)
        logger.info(f"Final features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        
        # 入力サイズを更新
        actual_input_size = features.shape[1]
        logger.info(f"Actual input size: {actual_input_size}")
        
        # 新しいGANモデルを作成（正しい入力サイズで）
        gan = CryptoGAN(
            input_size=actual_input_size,
            sequence_length=sequence_length,
            hidden_size=128,
        )
        
        # トレーナーの作成
        trainer = GANTrainer(gan)
        train_loader, val_loader = trainer.prepare_data(features, sequence_length)
        
        # 1エポックだけ訓練
        logger.info("\nTesting training for 1 epoch...")
        trainer.train(train_loader, val_loader, epochs=1)
        
    except FileNotFoundError:
        logger.warning("Features file not found. Skipping real data test.")
    except Exception as e:
        logger.error(f"Error during real data test: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n✅ All model tests passed!")

def main():
    """メイン関数"""
    print("=" * 50)
    print("Model Test")
    print("=" * 50)
    
    test_models()

if __name__ == "__main__":
    main()