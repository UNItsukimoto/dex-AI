#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
モデルの詳細診断 - なぜ予測精度が低いのか
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CryptoGAN
from src.data import HyperliquidDataLoader
from src.features import FeatureManager
from src.utils import get_logger

logger = get_logger(__name__)

class DetailedModelDiagnosis:
    """モデルの詳細診断"""
    
    def __init__(self):
        self.model_dir = Path('data/models')
        self.model = None
        self.metadata = None
        self.scaler = None
        
    def load_model_components(self):
        """モデルコンポーネントの読み込み"""
        # メタデータ
        with open(self.model_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
            
        # スケーラー
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # モデル
        self.model = CryptoGAN(
            input_size=self.metadata['input_size'],
            sequence_length=self.metadata['sequence_length']
        )
        
        # チェックポイント
        checkpoint_path = list((self.model_dir / 'checkpoints').glob('*_generator.pth'))[-1]
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.generator.load_state_dict(checkpoint)
            
        self.model.generator.eval()
        
    async def analyze_feature_importance(self):
        """特徴量の重要度分析"""
        # データの取得
        loader = HyperliquidDataLoader()
        feature_manager = FeatureManager()
        
        df = await loader.download_historical_data('BTC', '1h', days_back=30)
        df_features = feature_manager.create_all_features(df)
        
        # 数値特徴量のみ
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        df_numeric = df_features[numeric_columns].fillna(0)
        
        # 相関分析
        price_correlations = df_numeric.corrwith(df_numeric['close']).sort_values(ascending=False)
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 上位相関
        top_features = price_correlations.head(20)
        ax1.barh(range(len(top_features)), top_features.values)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index)
        ax1.set_xlabel('Correlation with Close Price')
        ax1.set_title('Top 20 Features Correlated with Price')
        ax1.grid(True, alpha=0.3)
        
        # 相関ヒートマップ
        important_features = ['close', 'returns', 'volume', 'volatility', 'rsi_14', 'trend_strength']
        available_features = [f for f in important_features if f in df_numeric.columns]
        
        corr_matrix = df_numeric[available_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('data/results/feature_importance_analysis.png')
        logger.info("Saved feature importance analysis")
        
        return price_correlations
        
    def analyze_prediction_pattern(self, num_samples=100):
        """予測パターンの分析"""
        predictions = []
        inputs = []
        
        # ランダムな入力で予測を生成
        for _ in range(num_samples):
            # ランダムノイズ
            noise = self.model.generate_noise(1)
            hidden = self.model.generator.init_hidden(1)
            
            with torch.no_grad():
                output, new_hidden, attention = self.model.generator(noise, hidden)
                predictions.append(output.squeeze().numpy())
                # ノイズを2次元に変換して保存
                inputs.append(noise.squeeze().numpy().flatten())
        
        predictions = np.array(predictions)
        inputs = np.array(inputs)
        
        logger.info(f"Inputs shape: {inputs.shape}, Predictions shape: {predictions.shape}")
        
        # 分析
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 入力vs出力の関係
        ax = axes[0, 0]
        # 入力の最初の要素と予測の関係をプロット
        if inputs.shape[1] > 0 and predictions.shape[1] > 3:
            ax.scatter(inputs[:, 0], predictions[:, 3], alpha=0.5)
            ax.set_xlabel('First Input Feature')
            ax.set_ylabel('Predicted Feature 3')
        else:
            # 代替プロット：予測の分布
            ax.hist(predictions[:, 0] if predictions.shape[1] > 0 else [0], bins=30)
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Frequency')
        ax.set_title('Input vs Output Relationship')
        
        # 2. 予測の分布
        ax = axes[0, 1]
        for i in range(min(5, predictions.shape[1])):
            ax.hist(predictions[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predictions')
        ax.legend()
        
        # 3. 予測の時系列パターン
        ax = axes[0, 2]
        for i in range(5):
            ax.plot(predictions[i, :20], label=f'Sample {i}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.set_title('Prediction Patterns')
        ax.legend()
        
        # 4. 入力の変化に対する感度
        ax = axes[1, 0]
        base_noise = self.model.generate_noise(1)
        sensitivities = []
        
        for scale in np.linspace(0.5, 2.0, 20):
            scaled_noise = base_noise * scale
            hidden = self.model.generator.init_hidden(1)
            
            with torch.no_grad():
                output, _, _ = self.model.generator(scaled_noise, hidden)
                # 出力の次元を確認
                if output.shape[1] > 3:
                    sensitivities.append(output.squeeze()[3].item())  # close価格
                else:
                    sensitivities.append(output.squeeze()[0].item())  # 最初の出力
        
        ax.plot(np.linspace(0.5, 2.0, 20), sensitivities)
        ax.set_xlabel('Input Scale')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Model Sensitivity to Input Scale')
        ax.grid(True, alpha=0.3)
        
        # 5. 予測の自己相関
        ax = axes[1, 1]
        sample_pred = predictions[0]
        autocorr = [np.corrcoef(sample_pred[:-i], sample_pred[i:])[0, 1] 
                   for i in range(1, min(20, len(sample_pred)))]
        ax.plot(range(1, len(autocorr)+1), autocorr)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Prediction Autocorrelation')
        ax.grid(True, alpha=0.3)
        
        # 6. 実際のデータとの比較
        ax = axes[1, 2]
        # 簡易的な統計比較
        ax.text(0.1, 0.8, f"Prediction Stats:", transform=ax.transAxes, fontsize=12, weight='bold')
        if predictions.shape[1] > 3:
            ax.text(0.1, 0.6, f"Mean: {np.mean(predictions[:, 3]):.2f}", transform=ax.transAxes)
            ax.text(0.1, 0.5, f"Std: {np.std(predictions[:, 3]):.2f}", transform=ax.transAxes)
            ax.text(0.1, 0.4, f"Min: {np.min(predictions[:, 3]):.2f}", transform=ax.transAxes)
            ax.text(0.1, 0.3, f"Max: {np.max(predictions[:, 3]):.2f}", transform=ax.transAxes)
        else:
            ax.text(0.1, 0.6, f"Output dimensions: {predictions.shape[1]}", transform=ax.transAxes)
            ax.text(0.1, 0.5, f"Mean: {np.mean(predictions):.2f}", transform=ax.transAxes)
            ax.text(0.1, 0.4, f"Std: {np.std(predictions):.2f}", transform=ax.transAxes)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/results/prediction_pattern_analysis.png')
        logger.info("Saved prediction pattern analysis")
        
    def check_training_data_quality(self):
        """訓練データの品質チェック"""
        # 訓練データの統計を確認
        try:
            # 保存されたデータがあれば読み込む
            processed_data_path = Path('data/processed/btc_features.csv')
            if processed_data_path.exists():
                df = pd.read_csv(processed_data_path, index_col=0)
                
                # 基本統計
                logger.info("\nTraining Data Statistics:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                
                # 欠損値の確認
                missing_ratio = df.isnull().sum() / len(df)
                high_missing = missing_ratio[missing_ratio > 0.1]
                if len(high_missing) > 0:
                    logger.warning(f"Features with >10% missing values: {high_missing.to_dict()}")
                
                # 定常性の確認（価格データ）
                if 'close' in df.columns:
                    from statsmodels.tsa.stattools import adfuller
                    result = adfuller(df['close'].dropna())
                    logger.info(f"ADF test p-value: {result[1]:.4f}")
                    if result[1] > 0.05:
                        logger.warning("Price data is non-stationary!")
                
        except Exception as e:
            logger.error(f"Failed to analyze training data: {e}")
            
    def provide_recommendations(self):
        """改善の推奨事項"""
        print("\n" + "="*60)
        print("MODEL IMPROVEMENT RECOMMENDATIONS")
        print("="*60)
        
        print("\n1. DATA PREPROCESSING:")
        print("   - Use differenced data (returns) instead of raw prices")
        print("   - Normalize each feature independently")
        print("   - Remove or impute missing values properly")
        print("   - Add more diverse features (sentiment, on-chain data)")
        
        print("\n2. MODEL ARCHITECTURE:")
        print("   - Try Transformer-based models instead of LSTM")
        print("   - Implement attention mechanism properly")
        print("   - Use residual connections")
        print("   - Consider ensemble methods")
        
        print("\n3. TRAINING IMPROVEMENTS:")
        print("   - Use proper train/validation split")
        print("   - Implement early stopping based on validation loss")
        print("   - Use learning rate scheduling")
        print("   - Train for price changes, not absolute prices")
        
        print("\n4. GAN SPECIFIC:")
        print("   - Implement Wasserstein GAN (WGAN-GP)")
        print("   - Use spectral normalization")
        print("   - Add auxiliary tasks (e.g., predict volatility)")
        print("   - Use conditional GAN with market regime")
        
        print("\n5. EVALUATION:")
        print("   - Focus on directional accuracy")
        print("   - Use proper backtesting with transaction costs")
        print("   - Test on different market conditions")
        print("   - Use walk-forward analysis")

async def main():
    """メイン診断関数"""
    diagnosis = DetailedModelDiagnosis()
    
    # モデルの読み込み
    logger.info("Loading model components...")
    diagnosis.load_model_components()
    
    # 特徴量の重要度分析
    logger.info("Analyzing feature importance...")
    correlations = await diagnosis.analyze_feature_importance()
    
    # 予測パターンの分析
    logger.info("Analyzing prediction patterns...")
    diagnosis.analyze_prediction_pattern()
    
    # 訓練データの品質チェック
    logger.info("Checking training data quality...")
    diagnosis.check_training_data_quality()
    
    # 推奨事項の提供
    diagnosis.provide_recommendations()
    
    logger.info("\nDiagnosis complete! Check the generated plots in data/results/")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())