#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import aiohttp
import json
import time

async def test_candles_api():
    """異なるパラメータでcandles APIをテスト"""
    base_url = "https://api.hyperliquid.xyz"
    
    # 現在時刻（ミリ秒）
    current_time = int(time.time() * 1000)
    one_day_ago = current_time - (24 * 60 * 60 * 1000)
    one_hour_ago = current_time - (60 * 60 * 1000)
    
    # テストするパラメータの組み合わせ
    test_params = [
        # 1. interval を数値にする
        {
            "type": "candleSnapshot", 
            "coin": "BTC",
            "interval": "60",  # 60分 = 1時間
            "startTime": one_day_ago,
            "endTime": current_time
        },
        # 2. req パラメータを使う
        {
            "type": "candleSnapshot",
            "req": {
                "coin": "BTC",
                "interval": "1h",
                "startTime": one_day_ago,
                "endTime": current_time
            }
        },
        # 3. 時間を秒単位にする
        {
            "type": "candleSnapshot",
            "coin": "BTC", 
            "interval": "1h",
            "startTime": int((time.time() - 86400)),  # 秒単位
            "endTime": int(time.time())
        },
        # 4. シンプルなパラメータ
        {
            "type": "candleSnapshot",
            "coin": "BTC",
            "interval": "1h"
        },
        # 5. 異なる型名を試す
        {
            "type": "candles",
            "coin": "BTC",
            "interval": "1h",
            "startTime": one_day_ago,
            "endTime": current_time
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, params in enumerate(test_params):
            print(f"\n=== Test {i+1} ===")
            print(f"Params: {json.dumps(params, indent=2)}")
            
            try:
                async with session.post(f"{base_url}/info", json=params) as response:
                    print(f"Status: {response.status}")
                    data = await response.text()
                    
                    if response.status == 200:
                        json_data = json.loads(data)
                        print(f"Success! Response type: {type(json_data)}")
                        if isinstance(json_data, list):
                            print(f"Number of candles: {len(json_data)}")
                            if len(json_data) > 0:
                                print(f"First candle: {json_data[0]}")
                        elif isinstance(json_data, dict):
                            print(f"Response keys: {list(json_data.keys())}")
                        break  # 成功したら終了
                    else:
                        print(f"Error response: {data[:200]}")
                        
            except Exception as e:
                print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_candles_api())