# -*- coding: utf-8 -*-
"""
GANトレーナー
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os

from .gan import CryptoGAN
from ..utils import get_logger

logger = get_logger(__name__)

class GANTrainer:
    """GANの訓練を管理するクラス"""
    
    def __init__(self,
                 model: CryptoGAN,
                 checkpoint_dir: str = "data/models/checkpoints"):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_history = {
            'd_loss': [],
            'g_loss': [],
            'd_acc_real': [],
            'd_acc_fake': [],
            'g_fool_rate': []
        }
        
    def prepare_data(self, 
                    features: np.ndarray,
                    sequence_length: int = 24,
                    train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """データの準備"""
        # データ型の確認と変換（np.objectの代わりにobjectを使用）
        if features.dtype == object:
            logger.warning("Features contain non-numeric data. Converting to float32...")
            try:
                features = features.astype(np.float32)
            except:
                raise ValueError("Cannot convert features to numeric type")
        
        # float32でない場合は変換
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        
        # NaNや無限大の処理
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or Inf values. Cleaning...")
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # シーケンスの作成
        sequences = []
        for i in range(len(features) - sequence_length):
            seq = features[i:i + sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences, dtype=np.float32)
        
        # 訓練・検証データに分割
        split_idx = int(len(sequences) * train_ratio)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # PyTorchのデータセットに変換
        train_dataset = TensorDataset(torch.FloatTensor(train_sequences))
        val_dataset = TensorDataset(torch.FloatTensor(val_sequences))
        
        # DataLoaderの作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )
        
        logger.info(f"Train sequences: {len(train_sequences)}")
        logger.info(f"Validation sequences: {len(val_sequences)}")
        logger.info(f"Sequence shape: {sequences[0].shape}")
        logger.info(f"Data type: {sequences.dtype}")
        
        return train_loader, val_loader
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 200,
              d_steps: int = 1,
              g_steps: int = 1,
              save_interval: int = 10):
        """GANの訓練"""
        logger.info(f"Starting GAN training for {epochs} epochs")
        
        for epoch in range(epochs):
            # 訓練モード
            self.model.generator.train()
            self.model.discriminator.train()
            
            d_losses = []
            g_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (real_data,) in enumerate(pbar):
                real_data = real_data.to(self.model.device)
                batch_size = real_data.size(0)
                
                # Discriminatorの訓練
                d_metrics = {}
                for _ in range(d_steps):
                    noise = self.model.generate_noise(batch_size)
                    d_metrics = self.model.train_discriminator(real_data, noise)
                    d_losses.append(d_metrics['d_loss'])
                
                # Generatorの訓練
                g_metrics = {}
                for _ in range(g_steps):
                    noise = self.model.generate_noise(batch_size)
                    g_metrics = self.model.train_generator(noise)
                    g_losses.append(g_metrics['g_loss'])
                
                # 進捗表示
                pbar.set_postfix({
                    'D_loss': f"{d_metrics['d_loss']:.4f}",
                    'G_loss': f"{g_metrics['g_loss']:.4f}",
                    'D_acc': f"{(d_metrics['d_acc_real'] + d_metrics['d_acc_fake'])/2:.2f}"
                })
            
            # エポックごとの平均
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            
            # 履歴に保存
            self.train_history['d_loss'].append(avg_d_loss)
            self.train_history['g_loss'].append(avg_g_loss)
            
            # 検証
            val_metrics = self.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"D_loss={avg_d_loss:.4f}, "
                f"G_loss={avg_g_loss:.4f}, "
                f"Val_loss={val_metrics['val_loss']:.4f}"
            )
            
            # チェックポイントの保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, val_metrics)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.generator.eval()
        self.model.discriminator.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for (real_data,) in val_loader:
                real_data = real_data.to(self.model.device)
                batch_size = real_data.size(0)
                
                # Discriminatorの検証
                real_output = self.model.discriminator(real_data)
                real_labels = torch.ones(batch_size, 1).to(self.model.device)
                val_loss = self.model.criterion(real_output, real_labels)
                
                val_losses.append(val_loss.item())
        
        return {
            'val_loss': np.mean(val_losses)
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """チェックポイントの保存"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"gan_epoch_{epoch}.pth"
        )
        
        self.model.generator.save_checkpoint(
            checkpoint_path.replace('.pth', '_generator.pth'),
            epoch=epoch,
            optimizer=self.model.optimizer_g,
            metrics=metrics
        )
        
        self.model.discriminator.save_checkpoint(
            checkpoint_path.replace('.pth', '_discriminator.pth'),
            epoch=epoch,
            optimizer=self.model.optimizer_d,
            metrics=metrics
        )
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """訓練履歴をプロット"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_history['d_loss']) + 1)
        
        ax.plot(epochs, self.train_history['d_loss'], label='Discriminator Loss', color='blue')
        ax.plot(epochs, self.train_history['g_loss'], label='Generator Loss', color='red')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('GAN Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training history plot to {save_path}")
        else:
            plt.show()
            
        plt.close()