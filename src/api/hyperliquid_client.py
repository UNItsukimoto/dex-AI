import aiohttp
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging
from datetime import datetime, timedelta

from ..utils import get_logger, config
from ..utils.decorators import async_timer, retry

logger = get_logger(__name__)

class HyperliquidClient:
    """Hyperliquid DEX APIクライアント"""
    
    def __init__(self):
        self.config = config.get_nested("hyperliquid", "config")
        self.base_url = self.config.get("base_url", "https://api.hyperliquid.xyz")
        self.ws_url = self.config.get("ws_url", "wss://api.hyperliquid.xyz/ws")
        self.session = None
        self._symbols_cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5分
        
    async def __aenter__(self):
        """非同期コンテキストマネージャーのエンター"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーのイグジット"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """セッションが存在することを確認"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    @retry(max_attempts=3, delay=1.0)
    async def _post_request(self, endpoint: str, payload: Dict) -> Dict:
        """POST リクエストの実行"""
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST request to {url} with payload: {payload}")
        
        try:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
    
    @async_timer
    async def get_all_mids(self) -> Dict[str, float]:
        """全ペアの中間価格を取得"""
        # キャッシュチェック
        cache_key = "all_mids"
        if self._is_cache_valid(cache_key):
            return self._symbols_cache[cache_key]
        
        payload = {"type": "allMids"}
        data = await self._post_request("/info", payload)
        
        # データ処理 - Hyperliquidの形式に対応
        mids = {}
        if isinstance(data, dict):
            for symbol, price_str in data.items():
                # @で始まるシンボルはスキップ（または含める場合はコメントアウト）
                if not symbol.startswith('@'):
                    try:
                        mids[symbol] = float(price_str)
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to parse price for {symbol}: {price_str}")
        
        # キャッシュ更新
        self._update_cache(cache_key, mids)
        
        logger.info(f"Retrieved {len(mids)} mid prices")
        return mids
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """オーダーブック取得"""
        payload = {
            "type": "l2Book",
            "coin": symbol
        }
        
        data = await self._post_request("/info", payload)
        
        order_book = {
            'symbol': symbol,
            'bids': [],
            'asks': [],
            'timestamp': data.get('time', int(time.time() * 1000))
        }
        
        # Hyperliquidの形式に対応
        if isinstance(data, dict) and 'levels' in data:
            levels = data['levels']
            if isinstance(levels, list) and len(levels) >= 2:
                # Bids処理
                bids_data = levels[0] if len(levels) > 0 else []
                for bid in bids_data[:depth]:
                    if isinstance(bid, dict) and 'px' in bid and 'sz' in bid:
                        order_book['bids'].append({
                            'price': float(bid['px']),
                            'size': float(bid['sz']),
                            'count': int(bid.get('n', 1))  # 注文数
                        })
                
                # Asks処理
                asks_data = levels[1] if len(levels) > 1 else []
                for ask in asks_data[:depth]:
                    if isinstance(ask, dict) and 'px' in ask and 'sz' in ask:
                        order_book['asks'].append({
                            'price': float(ask['px']),
                            'size': float(ask['sz']),
                            'count': int(ask.get('n', 1))
                        })
        
        return order_book
    
    async def get_candles(self, symbol: str, interval: str, 
                         lookback_days: int = 30) -> List[Dict]:
        """ローソク足データ取得"""
        end_time = int(time.time() * 1000)
        start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
        
        # Hyperliquidは req オブジェクトでパラメータをラップする必要がある
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time
            }
        }
        
        data = await self._post_request("/info", payload)
        
        candles = []
        if isinstance(data, list):
            for candle in data:
                if isinstance(candle, dict):
                    # Hyperliquidのキャンドル形式:
                    # t: 開始時刻, T: 終了時刻, s: シンボル, i: インターバル
                    # o: 始値, c: 終値, h: 高値, l: 安値, v: 出来高, n: 取引数
                    candles.append({
                        'timestamp': candle.get('t', 0),
                        'timestamp_end': candle.get('T', 0),
                        'symbol': candle.get('s', symbol),
                        'interval': candle.get('i', interval),
                        'open': float(candle.get('o', 0)),
                        'high': float(candle.get('h', 0)),
                        'low': float(candle.get('l', 0)),
                        'close': float(candle.get('c', 0)),
                        'volume': float(candle.get('v', 0)),
                        'trades': int(candle.get('n', 0))
                    })
        
        # タイムスタンプでソート
        candles.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Retrieved {len(candles)} candles for {symbol}")
        return candles
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """最近の取引履歴取得"""
        # まず req オブジェクトを試す
        payload = {
            "type": "trades",
            "req": {
                "coin": symbol
            }
        }
        
        try:
            data = await self._post_request("/info", payload)
        except aiohttp.ClientResponseError as e:
            if e.status == 422:
                # reqなしバージョンを試す
                payload = {"type": "trades", "coin": symbol}
                data = await self._post_request("/info", payload)
            else:
                raise
        
        trades = []
        trade_list = data if isinstance(data, list) else data.get('trades', [])
        
        for trade in trade_list[:limit]:
            if isinstance(trade, dict):
                trades.append({
                    'timestamp': trade.get('time', trade.get('t', 0)),
                    'price': float(trade.get('px', trade.get('p', 0))),
                    'size': float(trade.get('sz', trade.get('s', 0))),
                    'side': trade.get('side', 'unknown')
                })
        
        return trades
    
    async def get_funding_rates(self, symbol: str) -> Dict:
        """ファンディングレート取得"""
        payload = {
            "type": "fundingHistory",
            "req": {
                "coin": symbol,
                "startTime": int((time.time() - 86400) * 1000)  # 24時間前から
            }
        }
        
        try:
            data = await self._post_request("/info", payload)
        except aiohttp.ClientResponseError as e:
            if e.status == 422:
                # 別の形式を試す
                payload = {
                    "type": "fundingHistory", 
                    "coin": symbol,
                    "startTime": int((time.time() - 86400) * 1000)
                }
                data = await self._post_request("/info", payload)
            else:
                raise
        
        if data and isinstance(data, list) and len(data) > 0:
            latest = data[-1]
            return {
                'rate': float(latest.get('fundingRate', 0)),
                'timestamp': latest.get('time', 0),
                'premium': float(latest.get('premium', 0))
            }
        
        return {'rate': 0, 'timestamp': 0, 'premium': 0}
    
    async def get_open_interest(self, symbol: str) -> Dict:
        """オープンインタレスト取得"""
        payload = {
            "type": "openInterest",
            "coin": symbol
        }
        
        data = await self._post_request("/info", payload)
        
        return {
            'symbol': symbol,
            'open_interest': float(data.get('openInterest', 0)),
            'timestamp': int(time.time() * 1000)
        }
    
    async def get_universe(self) -> List[Dict]:
        """取引可能な全シンボル情報を取得"""
        payload = {"type": "metaAndAssetCtxs"}
        data = await self._post_request("/info", payload)
        
        universe = []
        # universeキーの確認
        asset_list = data.get('universe', data.get('assets', []))
        
        for asset in asset_list:
            if isinstance(asset, dict):
                universe.append({
                    'symbol': asset.get('name', ''),
                    'max_leverage': asset.get('maxLeverage', 1),
                    'only_isolated': asset.get('onlyIsolated', False)
                })
        
        return universe
    
    def _is_cache_valid(self, key: str) -> bool:
        """キャッシュが有効かチェック"""
        if key not in self._cache_timestamp:
            return False
        
        elapsed = time.time() - self._cache_timestamp[key]
        return elapsed < self.cache_ttl
    
    def _update_cache(self, key: str, data: Any):
        """キャッシュを更新"""
        self._symbols_cache[key] = data
        self._cache_timestamp[key] = time.time()