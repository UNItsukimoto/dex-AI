# -*- coding: utf-8 -*-
"""
CNN Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..utils import get_logger

logger = get_logger(__name__)

class CNNDiscriminator(BaseModel):
    """1D CNN-based Discriminator"""
    
    def __init__(self,
                 input_channels: int = 1,
                 sequence_length: int = 24,
                 conv_channels: list = [64, 128, 256],
                 kernel_sizes: list = [7, 5, 3],
                 dropout: float = 0.4):
        super(CNNDiscriminator, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            # Convolution
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2
                )
            )
            
            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            
            in_channels = out_channels
        
        # 最後の畳み込み層の出力サイズを計算
        conv_output_length = sequence_length
        for _ in conv_channels:
            conv_output_length = (conv_output_length + 1) // 2  # stride=2
            
        flatten_size = conv_channels[-1] * conv_output_length
        
        # 全結合層
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            output: Discrimination score
        """
        # (batch_size, sequence_length, features) -> (batch_size, features, sequence_length)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 全結合層
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        
        # 出力層（シグモイドなし - BCEWithLogitsLossを使用）
        x = self.fc3(x)
        
        return x