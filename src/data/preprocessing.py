# -*- coding: utf-8 -*-
"""
データ前処理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional

from ..utils import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """データの前処理を行うクラス"""
    
    def __init__(self, scaler_type: str = 'minmax'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの基本的な前処理"""
        # NaNの処理
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 無限大の値を除去
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # インデックスをソート
        df = df.sort_index()
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量を追加"""
        # リターン
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ボラティリティ
        df['volatility'] = df['returns'].rolling(window=24).std()
        
        # 価格位置
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # ボリューム関連
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=24).mean()
        
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      fit: bool = True) -> pd.DataFrame:
        """特徴量をスケーリング"""
        if self.scaler is None or fit:
            if self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # 価格データは除外してスケーリング
        price_columns = ['open', 'high', 'low', 'close']
        feature_columns = [col for col in df.columns if col not in price_columns]
        
        if fit:
            scaled_features = self.scaler.fit_transform(df[feature_columns])
            self.feature_columns = feature_columns
        else:
            scaled_features = self.scaler.transform(df[feature_columns])
        
        # スケーリングされたデータでDataFrameを作成
        scaled_df = pd.DataFrame(
            scaled_features,
            index=df.index,
            columns=feature_columns
        )
        
        # 価格データを追加
        for col in price_columns:
            if col in df.columns:
                scaled_df[col] = df[col]
        
        return scaled_df
    
    def create_sequences(self, 
                        df: pd.DataFrame, 
                        sequence_length: int = 24,
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """時系列シーケンスを作成"""
        data = df.values
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length + prediction_horizon - 1]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train_test_split(self, 
                        df: pd.DataFrame, 
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """訓練データとテストデータに分割"""
        split_index = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        return train_df, test_df