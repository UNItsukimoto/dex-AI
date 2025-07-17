# -*- coding: utf-8 -*-
"""
LSTMジェネレーター
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base_model import BaseModel
from ..utils import get_logger

logger = get_logger(__name__)

class LSTMGenerator(BaseModel):
    """LSTM-based Generator for time series prediction"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super(LSTMGenerator, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 出力層
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 正規化層
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # 初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state
            
        Returns:
            output: Predicted values
            hidden: Updated hidden state
            attention_weights: Attention weights
        """
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = lstm_out + attn_out
        
        # 最後のタイムステップの出力を使用
        # または全タイムステップの平均を使用
        if self.training:
            # 訓練時は全タイムステップの情報を使用
            output = torch.mean(lstm_out, dim=1)
        else:
            # 推論時は最後のタイムステップを使用
            output = lstm_out[:, -1, :]
        
        # 全結合層
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output, hidden, attention_weights
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """隠れ状態の初期化"""
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(self.device)
        
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(self.device)
        
        return (h0, c0)