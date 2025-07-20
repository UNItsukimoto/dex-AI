#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルデータ予測システム
Hyperliquid APIからのリアルタイムデータを使用した予測
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from hyperliquid_api_client import HyperliquidAPIClient
from simple_effective_2025_06 import SimpleEffective2025_06

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataPredictionSystem:
    """リアルデータ予測システム"""
    
    def __init__(self):
        self.api_client = None
        self.prediction_models = {}
        self.live_predictions = {}
        self.data_cache = {}
        
        # 対象銘柄
        self.target_symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
        
        # 予測更新間隔（秒）
        self.prediction_interval = 60  # 1分間隔
        
        # データ保存期間
        self.data_retention_hours = 168  # 1週間
        
    async def initialize(self):
        """システム初期化"""
        try:
            logger.info("Initializing Real Data Prediction System...")
            
            # API クライアント初期化
            self.api_client = HyperliquidAPIClient()
            await self.api_client.connect()
            
            # 予測モデル初期化
            await self._initialize_prediction_models()
            
            # 初期データ取得
            await self._fetch_initial_data()
            
            logger.info("Real Data Prediction System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def _initialize_prediction_models(self):
        """予測モデル初期化"""
        try:
            logger.info("Initializing prediction models...")
            
            for symbol in self.target_symbols:
                # シンプル効果的システムをベースに使用
                model = SimpleEffective2025_06()
                self.prediction_models[symbol] = model
                
                # 予測結果初期化
                self.live_predictions[symbol] = {
                    'probability': 0.5,
                    'confidence': 0.0,
                    'signal': 'HOLD',
                    'last_update': None,
                    'data_quality': 'unknown'
                }
            
            logger.info(f"Prediction models initialized for {len(self.target_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise
    
    async def _fetch_initial_data(self):
        """初期データ取得"""
        try:
            logger.info("Fetching initial data...")
            
            for symbol in self.target_symbols:
                # 過去24時間のデータ取得
                df = await self.api_client.get_recent_candles(symbol, '1h', 168)  # 1週間分
                
                if not df.empty:
                    self.data_cache[symbol] = df
                    logger.info(f"Initial data loaded for {symbol}: {len(df)} candles")
                else:
                    logger.warning(f"No initial data for {symbol}")
                    
        except Exception as e:
            logger.error(f"Initial data fetch error: {e}")
            raise
    
    def create_features_from_live_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ライブデータから特徴量作成"""
        try:
            if len(df) < 50:  # 最低限必要なデータ数
                logger.warning(f"Insufficient data for feature creation: {len(df)} candles")
                return pd.DataFrame()
            
            features_df = df.copy()
            
            # 1. 基本価格変動
            features_df['price_change_1h'] = df['close'].pct_change()
            features_df['price_change_2h'] = df['close'].pct_change(2)
            features_df['price_change_4h'] = df['close'].pct_change(4)
            features_df['price_change_8h'] = df['close'].pct_change(8)
            
            # 2. 移動平均（効果的期間のみ）
            for period in [5, 10, 20]:
                if len(df) >= period:
                    ma = df['close'].rolling(period).mean()
                    features_df[f'ma_ratio_{period}'] = df['close'] / ma
                    features_df[f'ma_slope_{period}'] = ma.diff() / ma
            
            # 3. RSI
            for period in [14, 21]:
                if len(df) >= period + 10:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # 4. ボラティリティ
            for period in [10, 20]:
                if len(df) >= period:
                    features_df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close']
            
            # 5. MACD
            if len(df) >= 30:
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                features_df['macd'] = ema12 - ema26
                features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
                features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # 6. ボリンジャーバンド位置
            for period in [20]:
                if len(df) >= period:
                    rolling_mean = df['close'].rolling(period).mean()
                    rolling_std = df['close'].rolling(period).std()
                    features_df[f'bb_position_{period}'] = (df['close'] - rolling_mean) / rolling_std
            
            # 7. 価格位置（レンジ内位置）
            for period in [10, 20]:
                if len(df) >= period:
                    high_max = df['high'].rolling(period).max()
                    low_min = df['low'].rolling(period).min()
                    features_df[f'price_position_{period}'] = (df['close'] - low_min) / (high_max - low_min)
            
            # 8. 短期ラグ特徴量
            for lag in [1, 2, 3]:
                features_df[f'price_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'change_lag_{lag}'] = features_df['price_change_1h'].shift(lag)
            
            # 9. 時間特徴量
            features_df['hour'] = df.index.hour
            features_df['is_morning'] = (df.index.hour < 12).astype(int)
            features_df['is_evening'] = (df.index.hour >= 18).astype(int)
            
            # ターゲット（実際の予測では使用しない）
            features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # データクリーニング
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            features_df = features_df.dropna()
            
            logger.debug(f"Created features: {len(features_df)} rows, {features_df.shape[1]} columns")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return pd.DataFrame()
    
    async def update_predictions(self):
        """予測を更新"""
        try:
            logger.info("Updating predictions...")
            
            for symbol in self.target_symbols:
                try:
                    # 最新データ取得
                    latest_df = await self.api_client.get_recent_candles(symbol, '1h', 100)
                    
                    if latest_df.empty:
                        logger.warning(f"No latest data for {symbol}")
                        continue
                    
                    # データキャッシュ更新
                    self.data_cache[symbol] = latest_df
                    
                    # 特徴量作成
                    features_df = self.create_features_from_live_data(latest_df)
                    
                    if features_df.empty:
                        logger.warning(f"No features created for {symbol}")
                        self.live_predictions[symbol]['data_quality'] = 'insufficient'
                        continue
                    
                    # 予測実行
                    prediction_result = await self._generate_prediction(symbol, features_df)
                    
                    # 結果を保存
                    self.live_predictions[symbol].update(prediction_result)
                    self.live_predictions[symbol]['last_update'] = datetime.now()
                    self.live_predictions[symbol]['data_quality'] = 'good'
                    
                    logger.info(f"{symbol} prediction updated: {prediction_result['probability']:.1%} ({prediction_result['signal']})")
                    
                except Exception as e:
                    logger.error(f"Prediction update error for {symbol}: {e}")
                    self.live_predictions[symbol]['data_quality'] = 'error'
                    continue
            
        except Exception as e:
            logger.error(f"Predictions update error: {e}")
    
    async def _generate_prediction(self, symbol: str, features_df: pd.DataFrame) -> Dict:
        """個別銘柄の予測生成"""
        try:
            # 特徴量とターゲット分離
            feature_cols = [col for col in features_df.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
            
            if len(feature_cols) == 0:
                logger.warning(f"No feature columns for {symbol}")
                return self._get_default_prediction()
            
            X = features_df[feature_cols]
            
            # 数値型のみ保持
            numeric_cols = []
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if not X[col].isnull().all():
                        numeric_cols.append(col)
                except:
                    continue
            
            if len(numeric_cols) == 0:
                logger.warning(f"No numeric features for {symbol}")
                return self._get_default_prediction()
            
            X = X[numeric_cols]
            X = X.dropna()
            
            if len(X) < 20:  # 最低限必要なサンプル数
                logger.warning(f"Insufficient samples for {symbol}: {len(X)}")
                return self._get_default_prediction()
            
            # 簡易予測（最新の価格トレンドベース）
            latest_features = X.iloc[-1:]
            
            # 価格変動から予測確率計算
            recent_changes = []
            for col in ['price_change_1h', 'price_change_2h', 'price_change_4h']:
                if col in latest_features.columns:
                    recent_changes.append(latest_features[col].iloc[0])
            
            if recent_changes:
                avg_change = np.mean([x for x in recent_changes if not np.isnan(x)])
                
                # シグモイド関数で確率に変換
                probability = 1 / (1 + np.exp(-avg_change * 50))  # スケール調整
                probability = max(0.1, min(0.9, probability))  # 範囲制限
                
                # 信頼度計算（データ品質ベース）
                confidence = min(0.9, len(X) / 100.0)
                
                # シグナル判定
                if probability > 0.6:
                    signal = 'BUY'
                elif probability < 0.4:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                return {
                    'probability': probability,
                    'confidence': confidence,
                    'signal': signal,
                    'features_count': len(numeric_cols),
                    'data_points': len(X)
                }
            else:
                return self._get_default_prediction()
                
        except Exception as e:
            logger.error(f"Prediction generation error for {symbol}: {e}")
            return self._get_default_prediction()
    
    def _get_default_prediction(self) -> Dict:
        """デフォルト予測結果"""
        return {
            'probability': 0.5,
            'confidence': 0.1,
            'signal': 'HOLD',
            'features_count': 0,
            'data_points': 0
        }
    
    def get_current_predictions(self) -> Dict:
        """現在の予測結果を取得"""
        return self.live_predictions.copy()
    
    def get_live_prices(self) -> Dict:
        """ライブ価格取得"""
        try:
            prices = {}
            for symbol in self.target_symbols:
                price = self.api_client.get_live_price(symbol)
                if price:
                    prices[symbol] = price
                elif symbol in self.data_cache and not self.data_cache[symbol].empty:
                    # キャッシュから最新価格取得
                    prices[symbol] = self.data_cache[symbol]['close'].iloc[-1]
                else:
                    prices[symbol] = None
            return prices
        except Exception as e:
            logger.error(f"Get live prices error: {e}")
            return {}
    
    def get_system_status(self) -> Dict:
        """システム状態取得"""
        try:
            status = {
                'connected': self.api_client.is_connected if self.api_client else False,
                'symbols_monitored': len(self.target_symbols),
                'predictions_available': len([p for p in self.live_predictions.values() if p['last_update']]),
                'last_update': None,
                'data_cache_status': {}
            }
            
            # 最後の更新時刻
            last_updates = [p['last_update'] for p in self.live_predictions.values() if p['last_update']]
            if last_updates:
                status['last_update'] = max(last_updates)
            
            # データキャッシュ状態
            for symbol in self.target_symbols:
                if symbol in self.data_cache:
                    df = self.data_cache[symbol]
                    status['data_cache_status'][symbol] = {
                        'rows': len(df),
                        'latest_time': df.index[-1] if not df.empty else None
                    }
                else:
                    status['data_cache_status'][symbol] = {'rows': 0, 'latest_time': None}
            
            return status
            
        except Exception as e:
            logger.error(f"Get system status error: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def start_live_prediction_loop(self):
        """ライブ予測ループ開始"""
        try:
            logger.info("Starting live prediction loop...")
            
            while True:
                start_time = datetime.now()
                
                # 予測更新
                await self.update_predictions()
                
                # 処理時間計算
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Prediction cycle completed in {processing_time:.2f}s")
                
                # 次の更新まで待機
                sleep_time = max(1, self.prediction_interval - processing_time)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Live prediction loop cancelled")
        except Exception as e:
            logger.error(f"Live prediction loop error: {e}")
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        try:
            if self.api_client:
                await self.api_client.disconnect()
            logger.info("Real Data Prediction System cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# テスト用関数
async def test_real_data_system():
    """リアルデータシステムのテスト"""
    print("=== Real Data Prediction System Test ===")
    
    system = RealDataPredictionSystem()
    
    try:
        # 初期化
        if await system.initialize():
            print("✓ System initialized successfully")
            
            # 初回予測実行
            await system.update_predictions()
            
            # 結果表示
            predictions = system.get_current_predictions()
            print("\nCurrent Predictions:")
            for symbol, pred in predictions.items():
                print(f"{symbol}: {pred['probability']:.1%} ({pred['signal']}) - Quality: {pred['data_quality']}")
            
            # ライブ価格表示
            prices = system.get_live_prices()
            print("\nLive Prices:")
            for symbol, price in prices.items():
                print(f"{symbol}: ${price:.2f}" if price else f"{symbol}: N/A")
            
            # システム状態表示
            status = system.get_system_status()
            print(f"\nSystem Status: {status}")
            
            print("\n✓ Test completed successfully")
        else:
            print("✗ System initialization failed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    finally:
        await system.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(test_real_data_system())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")