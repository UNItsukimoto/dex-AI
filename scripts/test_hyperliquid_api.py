#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperliquid APIのテストスクリプト
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加（dex-AIがルート）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.api import HyperliquidClient
from src.utils import get_logger

logger = get_logger(__name__)

async def test_api():
    """APIの基本機能をテスト"""
    async with HyperliquidClient() as client:
        try:
            # 1. 全シンボルの中間価格を取得
            logger.info("Testing get_all_mids...")
            mids = await client.get_all_mids()
            logger.info(f"Found {len(mids)} symbols")
            
            # BTCの価格を表示
            if 'BTC' in mids:
                logger.info(f"BTC mid price: ${mids['BTC']:,.2f}")
            
            # 2. BTCのオーダーブックを取得
            logger.info("\nTesting get_order_book...")
            orderbook = await client.get_order_book('BTC', depth=5)
            
            if orderbook['bids'] and orderbook['asks']:
                best_bid = orderbook['bids'][0]
                best_ask = orderbook['asks'][0]
                spread = best_ask['price'] - best_bid['price']
                
                logger.info(f"Best Bid: ${best_bid['price']:,.2f} ({best_bid['size']:.4f})")
                logger.info(f"Best Ask: ${best_ask['price']:,.2f} ({best_ask['size']:.4f})")
                logger.info(f"Spread: ${spread:.2f}")
            
            # 3. ローソク足データを取得
            logger.info("\nTesting get_candles...")
            candles = await client.get_candles('BTC', '1h', lookback_days=1)
            logger.info(f"Retrieved {len(candles)} candles")
            
            if candles:
                latest = candles[-1]
                logger.info(f"Latest candle - Close: ${latest['close']:,.2f}, "
                          f"Volume: {latest['volume']:.2f}")
            
            logger.info("\n✅ All API tests passed!")
            
        except Exception as e:
            logger.error(f"API test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン関数"""
    print("=" * 50)
    print("Hyperliquid API Test")
    print("=" * 50)
    
    # APIテスト
    asyncio.run(test_api())

if __name__ == "__main__":
    main()