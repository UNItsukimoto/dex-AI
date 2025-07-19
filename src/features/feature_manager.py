# -*- coding: utf-8 -*-
"""
特徴量マネージャー
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from .technical_indicators import TechnicalIndicators
from .market_microstructure import MarketMicrostructure
from .pattern_recognition import PatternRecognition
from ..utils import get_logger

logger = get_logger(__name__)

class FeatureManager:
    """すべての特徴量を管理するクラス"""
    
    def __init__(self):
        self.ti = TechnicalIndicators()
        self.mm = MarketMicrostructure()
        self.pr = PatternRecognition()
        self.feature_names = []
        
    def create_all_features(self, 
                          df: pd.DataFrame, 
                          orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """すべての特徴量を作成"""
        logger.info("Creating all features...")
        
        # 基本的な前処理
        df = self._preprocess_data(df)
        
        # テクニカル指標
        logger.info("Adding technical indicators...")
        df = self.ti.add_all_indicators(df)
        
        # マーケットマイクロストラクチャー
        logger.info("Adding market microstructure features...")
        df = self.mm.add_microstructure_features(df, orderbook_data)
        
        # パターン認識
        logger.info("Adding pattern recognition features...")
        df = self.pr.add_pattern_features(df)
        
        # 欠損値の処理
        df = self._handle_missing_values(df)
        
        # 特徴量名を保存
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの前処理"""
        # 基本的な特徴量が存在しない場合は追加
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の処理"""
        # 前方補完を優先、その後後方補完
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # それでも残る欠損値は0で埋める（または列ごとの平均値）
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
                
        return df
    
    def select_important_features(self, 
                                df: pd.DataFrame, 
                                target_col: str = 'close',
                                n_features: int = 50) -> List[str]:
        """重要な特徴量を選択（簡易版）"""
        # 相関係数による選択
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # 自身と完全相関のものを除外
        important_features = []
        for feature in correlations.index:
            if feature != target_col and len(important_features) < n_features:
                # 既に選択された特徴量との相関が高すぎないかチェック
                is_redundant = False
                for selected in important_features:
                    if abs(df[feature].corr(df[selected])) > 0.95:
                        is_redundant = True
                        break
                        
                if not is_redundant:
                    important_features.append(feature)
                    
        logger.info(f"Selected {len(important_features)} important features")
        
        return important_features