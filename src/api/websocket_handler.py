import asyncio
import json
import websockets
from typing import Callable, List, Dict, Optional
from ..utils import get_logger

logger = get_logger(__name__)

class HyperliquidWebSocket:
    """Hyperliquid WebSocket接続ハンドラー"""
    
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.websocket = None
        self.callbacks = {}
        self.subscriptions = set()
        self.running = False
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        
    async def connect(self):
        """WebSocket接続を確立"""
        attempt = 0
        while attempt < self.max_reconnect_attempts:
            try:
                self.websocket = await websockets.connect(self.ws_url)
                logger.info("WebSocket connected successfully")
                self.running = True
                return
            except Exception as e:
                attempt += 1
                logger.error(f"WebSocket connection failed (attempt {attempt}): {e}")
                if attempt < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay * attempt)
                else:
                    raise
    
    async def disconnect(self):
        """WebSocket接続を切断"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")
    
    async def subscribe(self, channel: str, params: Dict, callback: Callable):
        """チャンネルを購読"""
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": channel,
                **params
            }
        }
        
        # コールバックを登録
        channel_key = f"{channel}:{params.get('coin', 'all')}"
        self.callbacks[channel_key] = callback
        self.subscriptions.add(channel_key)
        
        # 購読メッセージを送信
        if self.websocket:
            await self.websocket.send(json.dumps(subscription))
            logger.info(f"Subscribed to {channel_key}")
    
    async def unsubscribe(self, channel: str, params: Dict):
        """チャンネルの購読を解除"""
        subscription = {
            "method": "unsubscribe",
            "subscription": {
                "type": channel,
                **params
            }
        }
        
        channel_key = f"{channel}:{params.get('coin', 'all')}"
        if channel_key in self.callbacks:
            del self.callbacks[channel_key]
            self.subscriptions.remove(channel_key)
        
        if self.websocket:
            await self.websocket.send(json.dumps(subscription))
            logger.info(f"Unsubscribed from {channel_key}")
    
    async def listen(self):
        """メッセージを受信して処理"""
        while self.running:
            try:
                if not self.websocket:
                    await self.connect()
                    # 再接続後、購読を復元
                    await self._restore_subscriptions()
                
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # データをコールバックに渡す
                await self._handle_message(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting to reconnect...")
                self.websocket = None
                await asyncio.sleep(self.reconnect_delay)
                
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, data: Dict):
        """受信メッセージを処理"""
        # メッセージタイプを判定
        if 'channel' in data:
            channel = data['channel']
            coin = data.get('data', {}).get('coin', 'all')
            channel_key = f"{channel}:{coin}"
            
            if channel_key in self.callbacks:
                callback = self.callbacks[channel_key]
                await callback(data['data'])
        elif 'type' in data:
            # 特殊なメッセージタイプの処理
            logger.debug(f"Received message type: {data['type']}")
    
    async def _restore_subscriptions(self):
        """再接続後に購読を復元"""
        for channel_key in list(self.subscriptions):
            # チャンネルキーを解析して再購読
            parts = channel_key.split(':')
            if len(parts) == 2:
                channel, coin = parts
                # 仮の再購読（実際のパラメータは保存しておく必要がある）
                logger.info(f"Restoring subscription: {channel_key}")