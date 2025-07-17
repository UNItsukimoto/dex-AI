# -*- coding: utf-8 -*-
"""
パターン認識
"""

import pandas as pd
import numpy as np
from scipy import signal

from ..utils import get_logger

logger = get_logger(__name__)

class PatternRecognition:
    """価格パターンの認識"""
    
    @staticmethod
    def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """パターン認識特徴量を追加"""
        df_copy = df.copy()
        
        # サポート・レジスタンスレベル
        df_copy = PatternRecognition.add_support_resistance(df_copy)
        
        # トレンド分析
        df_copy = PatternRecognition.add_trend_features(df_copy)
        
        # 価格パターン
        df_copy = PatternRecognition.add_price_patterns(df_copy)
        
        # フラクタル次元
        df_copy = PatternRecognition.add_fractal_dimension(df_copy)
        
        return df_copy
    
    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """サポート・レジスタンスレベルを検出"""
        # ローカルな高値・安値
        df['local_high'] = df['high'].rolling(window=window, center=True).max()
        df['local_low'] = df['low'].rolling(window=window, center=True).min()
        
        # 現在価格からの距離
        df['distance_to_high'] = (df['local_high'] - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['local_low']) / df['close']
        
        # ピボットポイント
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']  # レジスタンス1
        df['s1'] = 2 * df['pivot'] - df['high']  # サポート1
        
        return df
    
    @staticmethod
    def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """トレンド関連の特徴量を追加"""
        # 線形回帰によるトレンド
        for window in [24, 72, 168]:  # 1日、3日、7日
            x = np.arange(window)
            
            def calculate_slope(y):
                if len(y) < window:
                    return np.nan
                coef = np.polyfit(x, y, 1)
                return coef[0]
            
            df[f'trend_slope_{window}h'] = df['close'].rolling(window=window).apply(calculate_slope)
            
        # トレンドの強さ（R²）
        def calculate_r2(y):
            if len(y) < 24:
                return np.nan
            x = np.arange(len(y))
            coef = np.polyfit(x, y, 1)
            y_pred = np.polyval(coef, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        df['trend_strength'] = df['close'].rolling(window=24).apply(calculate_r2)
        
        # 価格の加速度
        df['price_acceleration'] = df['close'].diff().diff()
        
        return df
    
    @staticmethod
    def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """価格パターンを検出"""
        # ヘッドアンドショルダー検出（簡易版）
        window = 20
        
        # 局所的な極値を検出
        df['is_peak'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(-1))
        )
        df['is_valley'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(-1))
        )
        
        # 連続する高値・安値のカウント
        df['consecutive_higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(window=5).sum()
        df['consecutive_lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(window=5).sum()
        
        # ダブルトップ/ボトムの可能性
        df['double_top_signal'] = (
            (df['is_peak']) & 
            (df['is_peak'].shift(5, fill_value=False)) &
            (abs(df['high'] - df['high'].shift(5)) / df['high'] < 0.02)  # 2%以内
        )
        
        return df
    
    @staticmethod
    def add_fractal_dimension(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """フラクタル次元を計算（Hurstエクスポーント）"""
        def hurst_exponent(ts):
            """Hurstエクスポーントを計算"""
            if len(ts) < window:
                return np.nan
                
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            if len(tau) == 0:
                return np.nan
                
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['hurst_exponent'] = df['close'].rolling(window=window).apply(hurst_exponent)
        df['fractal_dimension'] = 2 - df['hurst_exponent']
        
        return df