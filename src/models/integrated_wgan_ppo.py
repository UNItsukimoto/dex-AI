#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合実装: WGAN-GP + PPO Hyperparameter Optimization
既存のモジュールを使用した修正版
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import os
import asyncio

# プロジェクトのルートディレクトリを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存のモジュールをインポート
from src.data.data_loader import HyperliquidDataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_manager import FeatureManager
from src.features.technical_indicators import TechnicalIndicators
from src.features.market_microstructure import MarketMicrostructure
from src.features.pattern_recognition import PatternRecognition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - **%(name)s** - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WGANGenerator(nn.Module):
    """WGAN-GP用のGenerator with LSTM architecture"""
    def __init__(self, input_dim: int, hidden_dim: int = 200, output_dim: int = 1, noise_dim: int = 20, sequence_length: int = 10):
        super().__init__()
        self.sequence_length = sequence_length
        self.noise_dim = noise_dim
        
        # LSTM層（stockpredictionaiスタイル）
        self.lstm = nn.LSTM(
            input_size=input_dim + noise_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output層
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            # Add sequence dimension
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
            noise = noise.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        combined = torch.cat([x, noise], dim=-1)
        lstm_out, _ = self.lstm(combined)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        return self.output_net(last_output)


class WGANCritic(nn.Module):
    """WGAN-GP用のCritic with 1D CNN architecture"""
    def __init__(self, input_dim: int, hidden_dim: int = 200, sequence_length: int = 10):
        super().__init__()
        self.sequence_length = sequence_length
        
        # 1D CNN層（stockpredictionaiスタイル）
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected層
        self.fc_net = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for CNN
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        conv_out = self.conv_net(x)
        conv_out = conv_out.squeeze(-1)  # Remove spatial dimension
        return self.fc_net(conv_out)


class IntegratedWGANPPOTrainer:
    """Integrated trainer combining WGAN-GP with existing project modules"""
    
    def __init__(
        self,
        output_dir: str = "models/wgan_ppo",
        device: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.scaler = StandardScaler()
        self.best_hyperparams = None
        self.generator = None
        self.critic = None
        
        # Initialize existing modules
        self.data_loader = HyperliquidDataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_manager = FeatureManager()
    
    def load_and_prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load and prepare data for training using existing data processing"""
        logger.info("Loading and preparing data...")
        
        # データディレクトリの確認と作成
        data_dir = project_root / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # まず、既存のデータが保存されているか確認
        saved_data_path = data_dir / "hyperliquid_btc_data.csv"
        
        if saved_data_path.exists():
            logger.info(f"Loading existing data from {saved_data_path}")
            df = pd.read_csv(saved_data_path)
            # timestampをインデックスに設定
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        else:
            logger.info("Fetching new data from Hyperliquid API...")
            # Hyperliquid APIからデータを取得
            df = asyncio.run(self.data_loader.download_historical_data(
                symbol="BTC",  # BTCに修正
                interval="1h",
                days_back=60
            ))
            
            if df is None or df.empty:
                logger.warning("Failed to fetch data from API, generating sample data...")
                # サンプルデータを生成
                df = self._generate_sample_data()
            
            # データを保存
            df.to_csv(saved_data_path)
            logger.info(f"Saved data to {saved_data_path}")
        
        # データの前処理
        logger.info("Preprocessing data...")
        df_processed = self.preprocessor.prepare_data(df)
        
        # 特徴量の作成
        logger.info("Creating features...")
        df_features = self.feature_manager.create_all_features(df_processed)
        
        # ターゲット変数を作成（次の価格変化率）
        df_features['target'] = df_features['close'].pct_change().shift(-1)
        
        # NaNを削除
        df_features = df_features.dropna()
        
        # 特徴量カラムを取得（数値カラムのみ）
        feature_columns = [col for col in df_features.columns 
                          if col not in ['target'] and df_features[col].dtype in ['float64', 'int64']]
        
        logger.info(f"Using {len(feature_columns)} features")
        
        # データの準備
        X = df_features[feature_columns].values
        y = df_features['target'].values
        
        # データを正規化
        X_scaled = self.scaler.fit_transform(X)
        
        # データを分割
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        X_val = X_scaled[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X_scaled[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # DataLoadersを作成
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
        
        # 初期バッチサイズ
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return train_loader, val_loader, test_loader, len(feature_columns)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """サンプルデータを生成（テスト用）"""
        logger.info("Generating sample data for testing...")
        
        # 日付範囲を作成
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1h')
        
        # ランダムな価格データを生成
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(1000) * 50,
            'high': close_prices + np.abs(np.random.randn(1000) * 100),
            'low': close_prices - np.abs(np.random.randn(1000) * 100),
            'close': close_prices,
            'volume': np.abs(np.random.randn(1000) * 1000000)
        }, index=dates)
        
        # 正しいOHLC関係を確保
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def train_wgan(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        input_dim: int,
        epochs: int = 100,
        n_critic: int = 5,
        lambda_gp: float = 10.0
    ) -> Dict[str, list]:
        """WGAN-GP訓練"""
        logger.info("Training WGAN-GP model...")
        
        # モデルの初期化
        self.generator = WGANGenerator(input_dim, noise_dim=20).to(self.device)
        self.critic = WGANCritic(input_dim).to(self.device)
        
        # オプティマイザ
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
        c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
        
        history = {'g_loss': [], 'c_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.generator.train()
            self.critic.train()
            
            g_losses = []
            c_losses = []
            
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                batch_size = features.size(0)
                
                # Train Critic
                for _ in range(n_critic):
                    c_optimizer.zero_grad()
                    
                    # Real data
                    real_data = torch.cat([features, targets.unsqueeze(1)], dim=1)
                    real_validity = self.critic(real_data)
                    
                    # Fake data
                    noise = torch.randn(batch_size, 20).to(self.device)
                    fake_targets = self.generator(features, noise)
                    fake_data = torch.cat([features, fake_targets.detach()], dim=1)
                    fake_validity = self.critic(fake_data)
                    
                    # Gradient penalty
                    gp = self._compute_gradient_penalty(real_data, fake_data)
                    
                    # Wasserstein loss + gradient penalty
                    c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
                    
                    c_loss.backward()
                    c_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, 20).to(self.device)
                fake_targets = self.generator(features, noise)
                fake_data = torch.cat([features, fake_targets], dim=1)
                fake_validity = self.critic(fake_data)
                
                g_loss = -torch.mean(fake_validity)
                
                # Add prediction loss
                pred_loss = nn.MSELoss()(fake_targets.squeeze(), targets)
                total_g_loss = g_loss + 0.1 * pred_loss
                
                total_g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                c_losses.append(c_loss.item())
            
            # Validation
            self.generator.eval()
            val_losses = []
            
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Average multiple predictions
                    preds = []
                    for _ in range(10):
                        noise = torch.randn(features.size(0), 20).to(self.device)
                        pred = self.generator(features, noise)
                        preds.append(pred)
                    
                    final_pred = torch.stack(preds).mean(dim=0)
                    val_loss = nn.MSELoss()(final_pred.squeeze(), targets)
                    val_losses.append(val_loss.item())
            
            history['g_loss'].append(np.mean(g_losses))
            history['c_loss'].append(np.mean(c_losses))
            history['val_loss'].append(np.mean(val_losses))
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"G Loss: {history['g_loss'][-1]:.4f}, "
                    f"C Loss: {history['c_loss'][-1]:.4f}, "
                    f"Val Loss: {history['val_loss'][-1]:.4f}"
                )
        
        return history
    
    def _compute_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        # Interpolate
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get critic output
        critic_interpolated = self.critic(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """訓練されたモデルを評価"""
        logger.info("Evaluating model performance...")
        
        self.generator.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # 複数のサンプルで予測して平均
                preds = []
                for _ in range(10):
                    noise = torch.randn(features.size(0), 20).to(self.device)
                    pred = self.generator(features, noise)
                    preds.append(pred)
                
                final_pred = torch.stack(preds).mean(dim=0)
                
                predictions.extend(final_pred.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # メトリクスを計算
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # 方向精度
        if len(predictions) > 1:
            pred_direction = predictions > 0
            actual_direction = actuals > 0
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            direction_accuracy = 0
        
        # 相関
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
        else:
            correlation = 0
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'correlation': float(correlation)
        }
        
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # メトリクスを保存
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def run_pipeline(self, training_epochs: int = 100):
        """パイプラインを実行"""
        logger.info("Starting WGAN-GP pipeline...")
        
        # データをロード
        train_loader, val_loader, test_loader, input_dim = self.load_and_prepare_data()
        
        # モデルを訓練
        training_history = self.train_wgan(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            epochs=training_epochs
        )
        
        # モデルを評価
        evaluation_metrics = self.evaluate_model(test_loader)
        
        # モデルを保存
        model_path = self.output_dir / 'wgan_model.pth'
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'scaler': self.scaler
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # サマリーレポートを生成
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'training_epochs': len(training_history['g_loss']),
            'final_metrics': evaluation_metrics
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Pipeline completed successfully!")
        return summary


def main():
    """Main execution function"""
    # Configuration
    config = {
        'output_dir': 'models/wgan_gp',
        'training_epochs': 10  # 短時間テスト用に減らす
    }
    
    # Create trainer
    trainer = IntegratedWGANPPOTrainer(
        output_dir=config['output_dir']
    )
    
    # Run pipeline
    results = trainer.run_pipeline(
        training_epochs=config['training_epochs']
    )
    
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()