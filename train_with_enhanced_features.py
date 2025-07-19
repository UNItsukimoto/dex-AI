#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拡張特徴量を使用した改良版訓練スクリプト
元のGitHubプロジェクトの要素を統合
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
from test_enhanced_features import EnhancedFeatureManager

logger = get_logger(__name__)

class AdvancedDataPreprocessor(DataPreprocessor):
    """高度なデータ前処理"""
    
    def prepare_enhanced_features(self, df, enhanced_features_df):
        """拡張特徴量を統合"""
        # 基本的なリターンベース特徴量
        df = self.prepare_return_based_features(df)
        
        # 拡張特徴量を結合
        df = pd.concat([df, enhanced_features_df], axis=1)
        
        # 重複列を削除
        df = df.loc[:, ~df.columns.duplicated()]
        
        # ターゲットラベルの作成
        df = self.create_target_labels(df)
        
        return df
    
    def prepare_return_based_features(self, df):
        """リターンベースの特徴量を準備（改良版）"""
        # 価格のリターンを計算
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 複数の時間枠でのリターン
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # ボリューム関連
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 価格の絶対値を削除（リターンベースに変換）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[f'{col}_return'] = df[col].pct_change()
                df = df.drop(col, axis=1)
        
        # 最初の行（NaN）を削除
        df = df.dropna()
        
        return df
    
    def create_target_labels(self, df, prediction_horizon=1):
        """予測ターゲットの作成（複数のターゲット）"""
        # 将来のリターンを予測
        df['target_return'] = df['returns'].shift(-prediction_horizon)
        
        # 方向性のラベル（上昇/下降）
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # マルチクラス分類（強い上昇/上昇/横ばい/下降/強い下降）
        thresholds = [-0.02, -0.005, 0.005, 0.02]  # -2%, -0.5%, 0.5%, 2%
        df['target_class'] = pd.cut(df['target_return'], 
                                   bins=[-np.inf] + thresholds + [np.inf],
                                   labels=[0, 1, 2, 3, 4])
        
        # 最後のN行を削除（ターゲットがないため）
        df = df[:-prediction_horizon]
        
        return df

async def train_enhanced_model(
    symbol: str = 'BTC',
    days_back: int = 90,  # より多くのデータ
    epochs: int = 200,    # より多くのエポック
    save_dir: str = 'data/models_v3_enhanced'
):
    """拡張特徴量を使用した改良版モデルの訓練"""
    
    logger.info(f"Starting enhanced training for {symbol} with extended features")
    
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
    
    # 2. 基本的な特徴量エンジニアリング
    logger.info("Creating basic features...")
    feature_manager = FeatureManager()
    df_features = feature_manager.create_all_features(df)
    
    # 3. 拡張特徴量の作成
    logger.info("Creating enhanced features...")
    enhanced_manager = EnhancedFeatureManager()
    
    # 相関資産の特徴量
    corr_features = await enhanced_manager.get_correlated_assets_features(symbol, '1h', days_back)
    
    # ARIMA特徴量
    df_features = enhanced_manager.add_arima_predictions(df_features.copy(), 'close')
    
    # フーリエ特徴量
    df_features = enhanced_manager.add_fourier_features(df_features, 'close')
    
    # 市場レジーム特徴量
    df_features = enhanced_manager.add_market_regime_features(df_features, 'close')
    
    # 相関資産の特徴量を結合
    # インデックスを合わせて結合
    corr_features = corr_features.reindex(df_features.index)
    df_features = pd.concat([df_features, corr_features], axis=1)
    
    # 4. データ前処理
    logger.info("Preprocessing data...")
    preprocessor = AdvancedDataPreprocessor(scaler_type='minmax')
    
    # リターンベースの特徴量に変換
    df_processed = preprocessor.prepare_enhanced_features(df_features, pd.DataFrame())
    
    # 数値列のみを選択
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    # ターゲット列を除外
    feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
    
    df_numeric = df_processed[feature_columns]
    
    # NaNと無限大の処理
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # スケーリング
    df_scaled = preprocessor.scale_features(df_numeric, fit=True)
    
    # ターゲットの準備
    target_returns = df_processed['target_return'].values
    target_directions = df_processed['target_direction'].values
    
    # NumPy配列に変換
    features = df_scaled.values.astype(np.float32)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Number of features: {features.shape[1]}")
    logger.info(f"Target shape: {target_returns.shape}")
    
    # 特徴量の統計情報
    logger.info("\nFeature statistics:")
    logger.info(f"Mean: {np.mean(features):.4f}")
    logger.info(f"Std: {np.std(features):.4f}")
    logger.info(f"Contains NaN: {np.any(np.isnan(features))}")
    
    # 5. モデルの作成
    logger.info("Creating enhanced model...")
    input_size = features.shape[1]
    sequence_length = 48  # より長いシーケンス
    
    gan = CryptoGAN(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=512,  # より大きなモデル
        learning_rate_g=0.0001,
        learning_rate_d=0.0001
    )
    
    # 6. 訓練データの準備
    logger.info("Preparing training data...")
    
    # シーケンスとターゲットの作成
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target_returns[i-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # 訓練/検証分割
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 7. 訓練
    logger.info("Training enhanced model...")
    
    # データローダーの作成
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    
    # トレーナーの初期化
    trainer = GANTrainer(gan, checkpoint_dir=str(checkpoint_dir))
    
    # 訓練の実行
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        d_steps=2,  # Discriminatorをより多く訓練
        g_steps=1,
        save_interval=20
    )
    
    # 8. メタデータの保存
    logger.info("Saving metadata...")
    
    metadata = {
        'symbol': symbol,
        'days_back': days_back,
        'epochs': epochs,
        'input_size': input_size,
        'sequence_length': sequence_length,
        'feature_columns': feature_columns,
        'num_features': len(feature_columns),
        'prediction_type': 'returns_enhanced',
        'training_date': datetime.now().isoformat(),
        'data_stats': {
            'mean_return': float(np.mean(target_returns)),
            'std_return': float(np.std(target_returns)),
            'positive_return_ratio': float(np.mean(target_directions))
        },
        'enhanced_features': {
            'correlated_assets': True,
            'arima': True,
            'fourier': True,
            'market_regime': True
        }
    }
    
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # スケーラーの保存
    import pickle
    with open(save_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    
    # 特徴量の重要度を保存
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0)
    })
    feature_importance.to_csv(save_path / 'feature_importance.csv', index=False)
    
    logger.info("Enhanced training completed!")
    logger.info(f"Model saved to {save_path}")
    
    return gan, preprocessor

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Crypto GAN model')
    parser.add_argument('--symbol', type=str, default='BTC', help='Crypto symbol')
    parser.add_argument('--days', type=int, default=90, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--save-dir', type=str, default='data/models_v3_enhanced', help='Save directory')
    
    args = parser.parse_args()
    
    asyncio.run(train_enhanced_model(
        symbol=args.symbol,
        days_back=args.days,
        epochs=args.epochs,
        save_dir=args.save_dir
    ))

if __name__ == "__main__":
    main()