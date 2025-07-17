# -*- coding: utf-8 -*-
"""
Hyperliquid データローダー
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple
import os

from ..api import HyperliquidClient
from ..utils import get_logger

logger = get_logger(__name__)

class HyperliquidDataLoader:
    """Hyperliquidからデータを取得して管理するクラス"""
    
    def __init__(self, cache_dir: str = "data/raw"):
        self.client = HyperliquidClient()
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """キャッシュディレクトリを作成"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    async def download_historical_data(self, 
                                     symbol: str, 
                                     interval: str = '1h',
                                     days_back: int = 30) -> pd.DataFrame:
        """履歴データのダウンロード"""
        logger.info(f"Downloading {days_back} days of {interval} data for {symbol}")
        
        async with self.client as client:
            # ローソク足データを取得
            candles = await client.get_candles(symbol, interval, days_back)
            
            if not candles:
                logger.warning(f"No candle data received for {symbol}")
                return pd.DataFrame()
            
            # DataFrameに変換
            df = pd.DataFrame(candles)
            
            # タイムスタンプをdatetimeに変換
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 不要なカラムを削除
            if 'timestamp_end' in df.columns:
                df.drop('timestamp_end', axis=1, inplace=True)
            if 'symbol' in df.columns:
                df.drop('symbol', axis=1, inplace=True)
            if 'interval' in df.columns:
                df.drop('interval', axis=1, inplace=True)
                
            logger.info(f"Downloaded {len(df)} candles for {symbol}")
            
            # キャッシュに保存
            self._save_to_cache(df, f"{symbol}_{interval}_{days_back}d")
            
            return df
    
    async def get_market_depth_snapshot(self, symbol: str) -> Dict:
        """現在のマーケットデプス（板情報）を取得"""
        async with self.client as client:
            orderbook = await client.get_order_book(symbol, depth=50)
            
            # マーケットデプス特徴量を計算
            features = self._calculate_depth_features(orderbook)
            
            return {
                'orderbook': orderbook,
                'features': features,
                'timestamp': datetime.now()
            }
    
    def _calculate_depth_features(self, orderbook: Dict) -> Dict:
        """オーダーブックから特徴量を計算"""
        features = {}
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return features
        
        # 買い圧・売り圧
        bid_volume = sum([b['size'] for b in bids])
        ask_volume = sum([a['size'] for a in asks])
        
        features['bid_volume'] = bid_volume
        features['ask_volume'] = ask_volume
        features['order_book_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # スプレッド
        best_bid = bids[0]['price'] if bids else 0
        best_ask = asks[0]['price'] if asks else 0
        features['spread'] = best_ask - best_bid
        features['spread_percentage'] = (features['spread'] / best_bid * 100) if best_bid > 0 else 0
        
        # 深度
        features['bid_depth_1pct'] = self._calculate_depth_at_percentage(bids, best_bid, 0.01, 'bid')
        features['ask_depth_1pct'] = self._calculate_depth_at_percentage(asks, best_ask, 0.01, 'ask')
        
        # 価格レベル
        features['bid_levels'] = len(bids)
        features['ask_levels'] = len(asks)
        
        return features
    
    def _calculate_depth_at_percentage(self, levels: List[Dict], 
                                     reference_price: float, 
                                     percentage: float, 
                                     side: str) -> float:
        """指定パーセンテージでの板の厚さを計算"""
        if not levels or reference_price <= 0:
            return 0
            
        if side == 'bid':
            threshold_price = reference_price * (1 - percentage)
            depth = sum([level['size'] for level in levels if level['price'] >= threshold_price])
        else:  # ask
            threshold_price = reference_price * (1 + percentage)
            depth = sum([level['size'] for level in levels if level['price'] <= threshold_price])
            
        return depth
    
    async def get_correlated_assets(self, 
                                  main_symbol: str = 'BTC',
                                  interval: str = '1h',
                                  days_back: int = 30) -> pd.DataFrame:
        """相関アセットのデータを取得"""
        correlated_symbols = {
            'BTC': ['ETH', 'SOL', 'BNB'],
            'ETH': ['BTC', 'ARB', 'OP', 'MATIC'],
            'SOL': ['BTC', 'ETH', 'AVAX']
        }
        
        symbols = correlated_symbols.get(main_symbol, ['BTC', 'ETH'])
        
        async with self.client as client:
            tasks = []
            for symbol in symbols:
                task = client.get_candles(symbol, interval, days_back)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # データを結合
        dfs = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, list) and result:
                df = pd.DataFrame(result)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                dfs[symbol] = df['close']
        
        if dfs:
            correlated_df = pd.DataFrame(dfs)
            logger.info(f"Retrieved data for {len(dfs)} correlated assets")
            return correlated_df
        else:
            logger.warning("No correlated asset data retrieved")
            return pd.DataFrame()
    
    async def get_all_market_data(self, symbol: str = 'BTC') -> Dict:
        """すべての市場データを取得"""
        logger.info(f"Fetching all market data for {symbol}")
        
        async with self.client as client:
            # 並列でデータを取得
            tasks = {
                'mids': client.get_all_mids(),
                'orderbook': client.get_order_book(symbol),
                'trades': client.get_trades(symbol, limit=100),
                'funding': client.get_funding_rates(symbol),
                'oi': client.get_open_interest(symbol)
            }
            
            results = {}
            for name, task in tasks.items():
                try:
                    results[name] = await task
                except Exception as e:
                    logger.error(f"Failed to get {name}: {e}")
                    results[name] = None
        
        return results
    
    def _save_to_cache(self, df: pd.DataFrame, filename: str):
        """データをキャッシュに保存"""
        filepath = os.path.join(self.cache_dir, f"{filename}.csv")
        df.to_csv(filepath)
        logger.debug(f"Saved data to {filepath}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータを読み込み"""
        filepath = os.path.join(self.cache_dir, f"{filename}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.debug(f"Loaded data from {filepath}")
            return df
        return None