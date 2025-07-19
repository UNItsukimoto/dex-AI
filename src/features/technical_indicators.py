# -*- coding: utf-8 -*-
"""
テクニカル指標の計算
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional

from ..utils import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """テクニカル指標を計算するクラス"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """すべてのテクニカル指標を追加"""
        df_copy = df.copy()
        
        # 価格関連の指標
        df_copy = TechnicalIndicators.add_moving_averages(df_copy)
        df_copy = TechnicalIndicators.add_bollinger_bands(df_copy)
        df_copy = TechnicalIndicators.add_rsi(df_copy)
        df_copy = TechnicalIndicators.add_macd(df_copy)
        
        # ボリューム関連の指標
        df_copy = TechnicalIndicators.add_volume_indicators(df_copy)
        
        # モメンタム指標
        df_copy = TechnicalIndicators.add_momentum_indicators(df_copy)
        
        # ボラティリティ指標
        df_copy = TechnicalIndicators.add_volatility_indicators(df_copy)
        
        logger.info(f"Added technical indicators. Total features: {len(df_copy.columns)}")
        
        return df_copy
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """移動平均を追加"""
        # 単純移動平均
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
        # 指数移動平均
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
        # 価格と移動平均の乖離率
        df['price_sma7_ratio'] = df['close'] / df['sma_7'] - 1
        df['price_sma21_ratio'] = df['close'] / df['sma_21'] - 1
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """ボリンジャーバンドを追加"""
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        
        df['bb_upper'] = rolling_mean + (rolling_std * num_std)
        df['bb_lower'] = rolling_mean - (rolling_std * num_std)
        df['bb_middle'] = rolling_mean
        
        # バンド幅
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 価格の位置（0-1の範囲）
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: list = [14, 21]) -> pd.DataFrame:
        """RSI（相対力指数）を追加"""
        for period in periods:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                close=df['close'], 
                window=period
            ).rsi()
            
        # RSIの変化率
        df['rsi_14_change'] = df['rsi_14'].diff()
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """MACDを追加"""
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # MACDヒストグラムの傾き
        df['macd_diff_change'] = df['macd_diff'].diff()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """ボリューム関連指標を追加"""
        # ボリューム移動平均
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # ボリュームレシオ
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # オンバランスボリューム（OBV）
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        # ボリューム加重平均価格（VWAP）
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # 価格とVWAPの乖離
        df['price_vwap_ratio'] = df['close'] / df['vwap'] - 1
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム指標を追加"""
        # ストキャスティクス
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        ).williams_r()
        
        # Rate of Change (ROC)
        for period in [10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ指標を追加"""
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range()
        
        # ATRレシオ
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Keltnerチャネル
        keltner = ta.volatility.KeltnerChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20
        )
        df['keltner_upper'] = keltner.keltner_channel_hband()
        df['keltner_lower'] = keltner.keltner_channel_lband()
        
        # 高値安値の変動率
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # リターンのローリング標準偏差
        df['volatility_24h'] = df['returns'].rolling(window=24).std()
        df['volatility_7d'] = df['returns'].rolling(window=168).std()  # 7日 = 168時間
        
        return df