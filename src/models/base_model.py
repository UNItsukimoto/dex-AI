# -*- coding: utf-8 -*-
"""
ベースモデルクラス
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

from ..utils import get_logger

logger = get_logger(__name__)

class BaseModel(nn.Module):
    """すべてのモデルの基底クラス"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def count_parameters(self) -> int:
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer: Optional[Any] = None, **kwargs):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        # 追加の情報を保存
        for key, value in kwargs.items():
            checkpoint[key] = value
            
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
    def load_checkpoint(self, filepath: str, optimizer: Optional[Any] = None) -> Dict:
        """チェックポイントを読み込み"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    def to_device(self, data):
        """データをデバイスに転送"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.to_device(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        else:
            return data