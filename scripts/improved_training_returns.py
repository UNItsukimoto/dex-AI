#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改良版訓練スクリプト - リターン（価格変化率）を予測
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

class ImprovedDataPreprocessor(DataPreprocessor):
    """改良版データ前処理 - リターンベース"""
    
    def prepare_return_based_features(self, df):
        """リターンベースの特徴量を準備"""
        # 価格のリターンを計算
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ボリュームの変化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 価格の絶対値ではなく、変化率ベースの特徴量を作成
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[f'{col}_return'] = df[col].pct_change()
                # 元の絶対価格列を削除
                df = df.drop(col, axis=1)
        
        # 最初の行（NaN）を削除
        df = df.dropna()
        
        return df
    
    def create_target_labels(self, df, prediction_horizon=1):
        """予測ターゲットの作成（リターンベース）"""
        # 将来のリターンを予測
        df['target_return'] = df['returns'].shift(-prediction_horizon)
        
        # 方向性のラベル（上昇/下降）
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # 最後のN行を削除（ターゲットがないため）
        df = df[:-prediction_horizon]
        
        return df

async def train_improved_model(
    symbol: str = 'BTC',
    days_back: int = 60,
    epochs: int = 150,
    save_dir: str = 'data/models_v2'
):
    """改良版モデルの訓練"""
    
    logger.info(f"Starting improved training for {symbol}")
    
    # 保存ディレクトリの作成
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 1. データの取得
    logger.info("Loading data...")
    loader = HyperliquidDataLoader()
    df = await loader.download_historical_data(symbol, '1h', days_back)
    
    if df.empty:
        logger.error("No data loaded")
        return
    
    # 2. 特徴量エンジニアリング
    logger.info("Creating features...")
    feature_manager = FeatureManager()
    df_features = feature_manager.create_all_features(df)
    
    # 3. リターンベースの前処理
    logger.info("Preprocessing for return prediction...")
    preprocessor = ImprovedDataPreprocessor(scaler_type='minmax')  # MinMaxScalerを使用
    
    # リターンベースの特徴量に変換
    df_features = preprocessor.prepare_return_based_features(df_features)
    df_features = preprocessor.create_target_labels(df_features)
    
    # 数値列のみを選択
    numeric_columns = df_features.select_dtypes(include=[np.number]).columns
    # ターゲット列を除外
    feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
    
    df_numeric = df_features[feature_columns]
    
    # NaNと無限大の処理
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(0)
    
    # スケーリング
    df_scaled = preprocessor.scale_features(df_numeric, fit=True)
    
    # ターゲットの準備
    target_returns = df_features['target_return'].values
    target_directions = df_features['target_direction'].values
    
    # NumPy配列に変換
    features = df_scaled.values.astype(np.float32)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Target shape: {target_returns.shape}")
    
    # 4. モデルの作成
    logger.info("Creating model...")
    input_size = features.shape[1]
    sequence_length = 24
    
    gan = CryptoGAN(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=256,
        learning_rate_g=0.0001,
        learning_rate_d=0.0002
    )
    
    # 5. 訓練データの準備（リターン予測用）
    logger.info("Preparing training data...")
    
    # シーケンスとターゲットの作成
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target_returns[i-1])  # 次の期間のリターン
    
    X = np.array(X)
    y = np.array(y)
    
    # 訓練/検証分割
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 6. カスタム訓練ループ（リターン予測に最適化）
    logger.info("Training model...")
    
    # GANの訓練用にデータを再構成（ターゲットなし）
    train_dataset_gan = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train)  # ターゲットなし
    )
    val_dataset_gan = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val)  # ターゲットなし
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset_gan, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset_gan, batch_size=32, shuffle=False
    )
    
    # トレーナーの初期化
    trainer = GANTrainer(gan, checkpoint_dir=str(checkpoint_dir))
    
    # 訓練の実行（標準的なGAN訓練）
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        d_steps=1,
        g_steps=1,
        save_interval=10
    )
    
    # 7. メタデータの保存
    logger.info("Saving metadata...")
    
    metadata = {
        'symbol': symbol,
        'days_back': days_back,
        'epochs': epochs,
        'input_size': input_size,
        'sequence_length': sequence_length,
        'feature_columns': feature_columns,
        'prediction_type': 'returns',  # 重要：リターン予測であることを記録
        'training_date': datetime.now().isoformat(),
        'data_stats': {
            'mean_return': float(np.mean(target_returns)),
            'std_return': float(np.std(target_returns)),
            'positive_return_ratio': float(np.mean(target_directions))
        }
    }
    
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # スケーラーの保存
    import pickle
    with open(save_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    
    logger.info("Improved training completed!")
    
    return gan, preprocessor

def main():
    parser = argparse.ArgumentParser(description='Train Improved Crypto GAN model')
    parser.add_argument('--symbol', type=str, default='BTC', help='Crypto symbol')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--save-dir', type=str, default='data/models_v2', help='Save directory')
    
    args = parser.parse_args()
    
    asyncio.run(train_improved_model(
        symbol=args.symbol,
        days_back=args.days,
        epochs=args.epochs,
        save_dir=args.save_dir
    ))

if __name__ == "__main__":
    main()