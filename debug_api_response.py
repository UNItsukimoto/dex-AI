#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

async def debug_api():
    """APIレスポンスの形式を確認"""
    base_url = "https://api.hyperliquid.xyz"
    
    async with aiohttp.ClientSession() as session:
        # 1. allMidsのテスト
        print("=== Testing allMids ===")
        payload = {"type": "allMids"}
        async with session.post(f"{base_url}/info", json=payload) as response:
            data = await response.json()
            print(f"Status: {response.status}")
            print(f"Response type: {type(data)}")
            print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            if isinstance(data, list) and len(data) > 0:
                print(f"First item: {data[0]}")
            elif isinstance(data, dict):
                # 最初のいくつかのキーと値を表示
                for i, (k, v) in enumerate(data.items()):
                    if i < 3:
                        print(f"  {k}: {v}")
            print(f"Full response (first 500 chars): {str(data)[:500]}")
        
        print("\n=== Testing l2Book ===")
        # 2. l2Bookのテスト
        payload = {"type": "l2Book", "coin": "BTC"}
        async with session.post(f"{base_url}/info", json=payload) as response:
            data = await response.json()
            print(f"Status: {response.status}")
            print(f"Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
                if 'levels' in data:
                    levels = data['levels']
                    print(f"Levels type: {type(levels)}")
                    if isinstance(levels, list) and len(levels) > 0:
                        print(f"First level type: {type(levels[0])}")
                        if len(levels) > 0 and isinstance(levels[0], list) and len(levels[0]) > 0:
                            print(f"First bid: {levels[0][0] if levels[0] else 'No bids'}")
            print(f"Full response (first 500 chars): {str(data)[:500]}")

if __name__ == "__main__":
    asyncio.run(debug_api())