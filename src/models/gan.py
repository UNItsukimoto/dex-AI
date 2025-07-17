# -*- coding: utf-8 -*-
"""
GAN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel
from .lstm_generator import LSTMGenerator
from .cnn_discriminator import CNNDiscriminator
from ..utils import get_logger

logger = get_logger(__name__)

class CryptoGAN(BaseModel):
    """暗号通貨予測のためのGANモデル"""
    
    def __init__(self,
                 input_size: int,
                 sequence_length: int = 24,
                 hidden_size: int = 256,
                 learning_rate_g: float = 0.0002,
                 learning_rate_d: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999):
        super(CryptoGAN, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # Generator
        self.generator = LSTMGenerator(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=input_size,  # 全特徴量を生成
            dropout=0.2,
            bidirectional=True
        ).to(self.device)
        
        # Discriminator
        self.discriminator = CNNDiscriminator(
            input_channels=input_size,
            sequence_length=sequence_length,
            conv_channels=[64, 128, 256],
            kernel_sizes=[7, 5, 3],
            dropout=0.4
        ).to(self.device)
        
        # 損失関数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # オプティマイザー
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            betas=(beta1, beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(beta1, beta2)
        )
        
        # ログ
        logger.info(f"Generator parameters: {self.generator.count_parameters():,}")
        logger.info(f"Discriminator parameters: {self.discriminator.count_parameters():,}")
        
    def train_discriminator(self, real_data: torch.Tensor, noise: torch.Tensor) -> Dict[str, float]:
        """Discriminatorの訓練"""
        batch_size = real_data.size(0)
        
        # 勾配をリセット
        self.optimizer_d.zero_grad()
        
        # 実データでの訓練
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # 生成データでの訓練
        hidden = self.generator.init_hidden(batch_size)
        fake_data, _, _ = self.generator(noise, hidden)
        fake_data = fake_data.view(batch_size, 1, -1).repeat(1, self.sequence_length, 1)
        
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # 総損失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        
        self.optimizer_d.step()
        
        # 精度の計算
        real_acc = (real_output > 0).float().mean()
        fake_acc = (fake_output < 0).float().mean()
        
        return {
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_acc_real': real_acc.item(),
            'd_acc_fake': fake_acc.item()
        }
    
    def train_generator(self, noise: torch.Tensor) -> Dict[str, float]:
        """Generatorの訓練"""
        batch_size = noise.size(0)
        
        # 勾配をリセット
        self.optimizer_g.zero_grad()
        
        # 生成データ
        hidden = self.generator.init_hidden(batch_size)
        fake_data, _, attention_weights = self.generator(noise, hidden)
        fake_data = fake_data.view(batch_size, 1, -1).repeat(1, self.sequence_length, 1)
        
        # Discriminatorを騙す
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_fool_rate': (fake_output > 0).float().mean().item()
        }
    
    def generate_noise(self, batch_size: int) -> torch.Tensor:
        """ノイズの生成"""
        noise = torch.randn(
            batch_size,
            self.sequence_length,
            self.input_size
        ).to(self.device)
        
        return noise
    
    def predict(self, input_sequence: torch.Tensor, num_predictions: int = 1) -> torch.Tensor:
        """将来の値を予測"""
        self.generator.eval()
        
        with torch.no_grad():
            predictions = []
            current_sequence = input_sequence.clone()
            
            for _ in range(num_predictions):
                hidden = self.generator.init_hidden(1)
                pred, hidden, _ = self.generator(current_sequence, hidden)
                predictions.append(pred)
                
                # シーケンスを更新
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
        
        self.generator.train()
        
        return torch.cat(predictions, dim=0)