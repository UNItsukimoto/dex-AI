#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperliquid API統合クライアント
リアルタイムデータ取得とWebSocket接続
"""

import asyncio
import websockets
import json
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable
import time
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperliquidAPIClient:
    """Hyperliquid API統合クライアント"""
    
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz"
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.session = None
        self.ws_connection = None
        self.is_connected = False
        self.subscribers = {}
        
        # 利用可能な銘柄
        self.available_symbols = [
            'BTC', 'ETH', 'SOL', 'AVAX', 'DOGE', 'MATIC', 
            'NEAR', 'APT', 'ARB', 'OP', 'SUI', 'SEI'
        ]
        
        # データ保存用
        self.live_data = {}
        self.candle_data = {}
        
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        await self.disconnect()
    
    async def connect(self):
        """API接続を初期化"""
        try:
            # HTTP セッション作成（接続プール、タイムアウト設定）
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=10
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'HyperliquidAPIClient/1.0'}
            )
            
            # WebSocket接続（再接続機能付き）
            await self._connect_websocket()
            
            self.is_connected = True
            logger.info("Hyperliquid API connected successfully")
            
            # WebSocketメッセージ処理を開始
            asyncio.create_task(self._handle_ws_messages())
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            await self._cleanup_connections()
            raise
    
    async def _connect_websocket(self, max_retries: int = 3):
        """WebSocket接続（再接続機能付き）"""
        for attempt in range(max_retries):
            try:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                self.ws_connection = await websockets.connect(
                    self.ws_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=2**20,  # 1MB max message size
                    compression=None  # 圧縮無効化でパフォーマンス向上
                )
                
                logger.info(f"WebSocket connected on attempt {attempt + 1}")
                return
                
            except Exception as e:
                logger.warning(f"WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数バックオフ
    
    async def _cleanup_connections(self):
        """接続のクリーンアップ"""
        try:
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.close()
            if self.session and not self.session.closed:
                await self.session.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def disconnect(self):
        """API接続を切断"""
        try:
            self.is_connected = False
            await self._cleanup_connections()
            logger.info("Hyperliquid API disconnected")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    async def get_exchange_meta(self) -> Dict:
        """取引所メタデータ取得"""
        try:
            url = f"{self.base_url}/info"
            payload = {"type": "meta"}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Exchange metadata retrieved successfully")
                    return data
                else:
                    logger.error(f"Meta request failed: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Get meta error: {e}")
            return {}
    
    async def get_all_mids(self) -> Dict:
        """全銘柄の中間価格取得"""
        try:
            url = f"{self.base_url}/info"
            payload = {"type": "allMids"}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"All mids retrieved: {len(data)} symbols")
                    return data
                else:
                    logger.error(f"All mids request failed: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Get all mids error: {e}")
            return {}
    
    async def get_user_state(self, user_address: str) -> Dict:
        """ユーザー状態取得"""
        try:
            url = f"{self.base_url}/info"
            payload = {
                "type": "clearinghouseState",
                "user": user_address
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("User state retrieved successfully")
                    return data
                else:
                    logger.error(f"User state request failed: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Get user state error: {e}")
            return {}
    
    async def get_candles(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[Dict]:
        """ローソク足データ取得"""
        try:
            url = f"{self.base_url}/info"
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Candles retrieved for {symbol}: {len(data)} bars")
                    return data
                else:
                    logger.error(f"Candles request failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Get candles error: {e}")
            return []
    
    async def get_recent_candles(self, symbol: str, interval: str = "1h", count: int = 100) -> pd.DataFrame:
        """最近のローソク足データをDataFrameで取得"""
        try:
            # 終了時刻を現在時刻に設定
            end_time = int(time.time() * 1000)
            
            # インターバルに応じた開始時刻計算
            interval_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }.get(interval, 60)
            
            start_time = end_time - (count * interval_minutes * 60 * 1000)
            
            # ローソク足データ取得
            candles = await self.get_candles(symbol, interval, start_time, end_time)
            
            if not candles:
                logger.warning(f"No candle data received for {symbol}")
                return pd.DataFrame()
            
            # DataFrameに変換
            df_data = []
            for candle in candles:
                if isinstance(candle, dict) and 't' in candle:
                    df_data.append({
                        'timestamp': pd.to_datetime(candle['t'], unit='ms'),
                        'open': float(candle['o']),
                        'high': float(candle['h']),
                        'low': float(candle['l']),
                        'close': float(candle['c']),
                        'volume': float(candle['v']) if 'v' in candle else 0.0
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"Converted {len(df)} candles to DataFrame for {symbol}")
                return df
            else:
                logger.warning(f"No valid candle data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Get recent candles error: {e}")
            return pd.DataFrame()
    
    async def subscribe_to_all_mids(self, callback: Optional[Callable] = None):
        """全銘柄中間価格の購読"""
        try:
            if not self.is_connected:
                raise ConnectionError("Not connected to WebSocket")
            
            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "allMids"
                }
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            
            if callback:
                self.subscribers['allMids'] = callback
                
            logger.info("Subscribed to all mids")
            
        except Exception as e:
            logger.error(f"Subscribe to all mids error: {e}")
    
    async def subscribe_to_trades(self, symbol: str, callback: Optional[Callable] = None):
        """個別銘柄の取引データ購読"""
        try:
            if not self.is_connected:
                raise ConnectionError("Not connected to WebSocket")
            
            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "trades",
                    "coin": symbol
                }
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            
            if callback:
                self.subscribers[f'trades_{symbol}'] = callback
                
            logger.info(f"Subscribed to trades for {symbol}")
            
        except Exception as e:
            logger.error(f"Subscribe to trades error: {e}")
    
    async def _handle_ws_messages(self):
        """WebSocketメッセージ処理（自動再接続機能付き）"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while self.is_connected and reconnect_attempts < max_reconnect_attempts:
            try:
                if self.ws_connection.closed:
                    logger.warning("WebSocket connection closed, attempting reconnect...")
                    await self._connect_websocket()
                    reconnect_attempts += 1
                    continue
                    
                # メッセージ受信（タイムアウト付き）
                try:
                    message = await asyncio.wait_for(
                        self.ws_connection.recv(), 
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("WebSocket message timeout, sending ping...")
                    await self.ws_connection.ping()
                    continue
                    
                # JSONパース（エラーハンドリング強化）
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    continue
                
                # メッセージタイプに応じて処理
                if 'channel' in data:
                    await self._process_channel_message(data)
                elif 'subscription' in data:
                    await self._process_subscription_update(data)
                
                # 成功時は再接続カウンターリセット
                reconnect_attempts = 0
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed by server")
                if self.is_connected:
                    await asyncio.sleep(min(2 ** reconnect_attempts, 30))  # 指数バックオフ
                    reconnect_attempts += 1
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await asyncio.sleep(1)
                
        if reconnect_attempts >= max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, giving up")
            
        self.is_connected = False
    
    async def _process_channel_message(self, data: Dict):
        """チャンネルメッセージ処理"""
        try:
            channel = data.get('channel')
            
            if channel == 'allMids':
                # 全銘柄中間価格更新
                mids_data = data.get('data', {})
                self.live_data['mids'] = mids_data
                self.live_data['last_update'] = datetime.now()
                
                # コールバック実行
                if 'allMids' in self.subscribers:
                    await self.subscribers['allMids'](mids_data)
                    
            elif channel.startswith('trades'):
                # 取引データ更新
                symbol = data.get('data', {}).get('coin', 'unknown')
                trades = data.get('data', {}).get('trades', [])
                
                if f'trades_{symbol}' not in self.live_data:
                    self.live_data[f'trades_{symbol}'] = []
                    
                self.live_data[f'trades_{symbol}'].extend(trades)
                
                # 最新1000件のみ保持
                self.live_data[f'trades_{symbol}'] = self.live_data[f'trades_{symbol}'][-1000:]
                
                # コールバック実行
                callback_key = f'trades_{symbol}'
                if callback_key in self.subscribers:
                    await self.subscribers[callback_key](trades)
                    
        except Exception as e:
            logger.error(f"Process channel message error: {e}")
    
    async def _process_subscription_update(self, data: Dict):
        """購読更新処理"""
        try:
            subscription_type = data.get('subscription', {}).get('type')
            logger.debug(f"Subscription update: {subscription_type}")
            
        except Exception as e:
            logger.error(f"Process subscription update error: {e}")
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """ライブ価格取得"""
        try:
            if 'mids' in self.live_data:
                price_str = self.live_data['mids'].get(symbol)
                if price_str:
                    return float(price_str)
            return None
            
        except Exception as e:
            logger.error(f"Get live price error: {e}")
            return None
    
    def get_live_data_summary(self) -> Dict:
        """ライブデータサマリー取得"""
        try:
            summary = {
                'connected': self.is_connected,
                'last_update': self.live_data.get('last_update'),
                'symbols_count': len(self.live_data.get('mids', {})),
                'available_symbols': self.available_symbols
            }
            
            # 価格データがある場合
            if 'mids' in self.live_data:
                summary['sample_prices'] = {
                    symbol: self.live_data['mids'].get(symbol, 'N/A')
                    for symbol in ['BTC', 'ETH', 'SOL']
                    if symbol in self.live_data['mids']
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Get live data summary error: {e}")
            return {'connected': False, 'error': str(e)}

# テスト用の非同期関数
async def test_api_client():
    """APIクライアントのテスト"""
    print("=== Hyperliquid API Client Test ===")
    
    async with HyperliquidAPIClient() as client:
        print(f"Connected: {client.is_connected}")
        
        # メタデータ取得テスト
        print("\n1. Getting exchange metadata...")
        meta = await client.get_exchange_meta()
        if meta:
            print(f"Universe size: {len(meta.get('universe', []))}")
        
        # 全銘柄価格取得テスト
        print("\n2. Getting all mids...")
        mids = await client.get_all_mids()
        if mids:
            btc_price = mids.get('BTC', 'N/A')
            eth_price = mids.get('ETH', 'N/A')
            print(f"BTC: ${btc_price}, ETH: ${eth_price}")
        
        # ローソク足データ取得テスト
        print("\n3. Getting recent candles for BTC...")
        btc_df = await client.get_recent_candles('BTC', '1h', 24)
        if not btc_df.empty:
            print(f"Retrieved {len(btc_df)} candles")
            print(f"Latest price: ${btc_df['close'].iloc[-1]:.2f}")
            print(f"24h change: {((btc_df['close'].iloc[-1] / btc_df['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # WebSocket購読テスト
        print("\n4. Testing WebSocket subscription...")
        
        async def price_callback(data):
            btc_price = data.get('BTC', 'N/A')
            print(f"Live BTC price: ${btc_price}")
        
        await client.subscribe_to_all_mids(price_callback)
        
        # 5秒間リアルタイムデータを受信
        print("Receiving live data for 5 seconds...")
        await asyncio.sleep(5)
        
        # サマリー表示
        summary = client.get_live_data_summary()
        print(f"\nSummary: {summary}")

if __name__ == "__main__":
    # テスト実行
    try:
        asyncio.run(test_api_client())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")