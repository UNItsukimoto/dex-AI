# -*- coding: utf-8 -*-
"""
市場マイクロストラクチャー分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from ..utils import get_logger

logger = get_logger(__name__)

class MarketMicrostructure:
    """市場マイクロストラクチャーの特徴量を計算"""
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame, 
                                   orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """マイクロストラクチャー特徴量を追加"""
        df_copy = df.copy()
        
        # 価格効率性
        df_copy = MarketMicrostructure.add_price_efficiency(df_copy)
        
        # 流動性指標
        df_copy = MarketMicrostructure.add_liquidity_metrics(df_copy)
        
        # 取引強度
        df_copy = MarketMicrostructure.add_trade_intensity(df_copy)
        
        # ティックサイズ分析
        df_copy = MarketMicrostructure.add_tick_analysis(df_copy)
        
        return df_copy
    
    @staticmethod
    def add_price_efficiency(df: pd.DataFrame) -> pd.DataFrame:
        """価格効率性指標を追加"""
        # 価格の自己相関
        for lag in [1, 5, 10]:
            df[f'price_autocorr_{lag}'] = df['close'].rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Variance Ratio Test的な指標
        df['variance_ratio_5'] = (
            df['close'].pct_change(5).rolling(window=20).var() / 
            (5 * df['close'].pct_change().rolling(window=20).var())
        )
        
        # 価格発見効率
        df['price_discovery'] = 1 - abs(df['variance_ratio_5'] - 1)
        
        return df
    
    @staticmethod
    def add_liquidity_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """流動性指標を追加"""
        # Amihudの非流動性指標
        df['amihud_illiquidity'] = abs(df['returns']) / (df['volume'] + 1e-10)
        df['amihud_illiquidity_ma'] = df['amihud_illiquidity'].rolling(window=24).mean()
        
        # Kyle's Lambda (簡易版)
        df['kyle_lambda'] = abs(df['returns']) / np.sqrt(df['volume'] + 1e-10)
        
        # 実効スプレッド推定
        df['high_low_spread'] = 2 * np.sqrt(abs(
            np.log(df['high'] / df['close']) * np.log(df['low'] / df['close'])
        ))
        
        # ボリュームの集中度
        df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=24).sum()
        
        return df
    
    @staticmethod
    def add_trade_intensity(df: pd.DataFrame) -> pd.DataFrame:
        """取引強度指標を追加"""
        # 取引頻度
        df['trade_intensity'] = df['trades'] / df['trades'].rolling(window=24).mean()
        
        # 平均取引サイズ
        df['avg_trade_size'] = df['volume'] / (df['trades'] + 1)
        df['avg_trade_size_ma'] = df['avg_trade_size'].rolling(window=24).mean()
        
        # 大口取引の割合（推定）
        df['large_trade_ratio'] = df['avg_trade_size'] / df['avg_trade_size_ma']
        
        # ボリュームクロック
        df['volume_clock'] = df['volume'].cumsum()
        df['volume_time'] = df['volume_clock'] / df['volume'].rolling(window=168).sum()
        
        return df
    
    @staticmethod
    def add_tick_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """ティック分析"""
        # 価格変化の符号
        df['price_change_sign'] = np.sign(df['close'].diff())
        
        # アップティック/ダウンティックの比率
        df['uptick_ratio'] = df['price_change_sign'].rolling(window=24).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # 連続同方向変化
        df['consecutive_moves'] = df['price_change_sign'].groupby(
            (df['price_change_sign'] != df['price_change_sign'].shift()).cumsum()
        ).cumcount() + 1
        
        # ティックサイズに対する価格変化
        tick_size = 1  # Hyperliquidのティックサイズ（要確認）
        df['tick_ratio'] = abs(df['close'].diff()) / tick_size
        
        return df
    
    @staticmethod
    def calculate_orderbook_features(orderbook: Dict) -> Dict:
        """オーダーブックから特徴量を計算"""
        features = {}
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return features
        
        # 価格レベルごとの分析
        bid_prices = [b['price'] for b in bids]
        bid_sizes = [b['size'] for b in bids]
        ask_prices = [a['price'] for a in asks]
        ask_sizes = [a['size'] for a in asks]
        
        # 深度の偏り
        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)
        features['depth_imbalance'] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        # 加重平均価格
        if total_bid_size > 0:
            features['weighted_bid_price'] = sum(p * s for p, s in zip(bid_prices, bid_sizes)) / total_bid_size
        if total_ask_size > 0:
            features['weighted_ask_price'] = sum(p * s for p, s in zip(ask_prices, ask_sizes)) / total_ask_size
            
        # 板の形状
        features['bid_slope'] = np.polyfit(range(len(bid_sizes[:10])), bid_sizes[:10], 1)[0] if len(bid_sizes) >= 10 else 0
        features['ask_slope'] = np.polyfit(range(len(ask_sizes[:10])), ask_sizes[:10], 1)[0] if len(ask_sizes) >= 10 else 0
        
        return features