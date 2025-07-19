#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
過去の異なる期間のデータをダウンロード
"""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz"
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_period(self, symbol: str, start_date: datetime, end_date: datetime, 
                            period_name: str, interval: str = "1h"):
        """特定期間のデータをダウンロード"""
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/info"
                start_time = int(start_date.timestamp() * 1000)
                end_time = int(end_date.timestamp() * 1000)
                
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": interval,
                        "startTime": start_time,
                        "endTime": end_time
                    }
                }
                
                logger.info(f"Downloading {period_name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            # データフレームに変換
                            df = pd.DataFrame(data)
                            
                            # カラム名の正規化
                            column_mapping = {
                                't': 'timestamp',
                                'T': 'timestamp_end',
                                's': 'symbol',
                                'i': 'interval',
                                'o': 'open',
                                'c': 'close',
                                'h': 'high',
                                'l': 'low',
                                'v': 'volume',
                                'n': 'trades'
                            }
                            df = df.rename(columns=column_mapping)
                            
                            # タイムスタンプを日時に変換
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            
                            # 数値型に変換
                            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                            for col in numeric_cols:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # ファイル保存
                            filename = f"{symbol}_{interval}_{period_name}.csv"
                            filepath = self.data_dir / filename
                            df.to_csv(filepath)
                            
                            logger.info(f"Saved {len(df)} candles to {filename}")
                            return df
                        else:
                            logger.warning(f"No data available for {period_name}")
                            return None
                    else:
                        logger.error(f"API error: Status {response.status}")
                        return None
                        
            except Exception as e:
                logger.error(f"Error downloading {period_name}: {e}")
                return None
    
    async def download_multiple_periods(self):
        """複数の異なる期間のデータをダウンロード"""
        symbol = "BTC"
        
        # ダウンロードする期間のリスト
        periods = [
            # 2025年の各月（最新）
            ("2025_07", datetime(2025, 7, 1), datetime(2025, 7, 19)),
            ("2025_06", datetime(2025, 6, 1), datetime(2025, 6, 30)),
            ("2025_05", datetime(2025, 5, 1), datetime(2025, 5, 31)),
            ("2025_04", datetime(2025, 4, 1), datetime(2025, 4, 30)),
            
            # 2024年の四半期ごと
            ("2024_Q4", datetime(2024, 10, 1), datetime(2024, 12, 31)),
            ("2024_Q3", datetime(2024, 7, 1), datetime(2024, 9, 30)),
            ("2024_Q2", datetime(2024, 4, 1), datetime(2024, 6, 30)),
            ("2024_Q1", datetime(2024, 1, 1), datetime(2024, 3, 31)),
            
            # 2023年の半期ごと
            ("2023_H2", datetime(2023, 7, 1), datetime(2023, 12, 31)),
            ("2023_H1", datetime(2023, 1, 1), datetime(2023, 6, 30)),
            
            # 特定のイベント期間
            ("Bull_Run_2024", datetime(2024, 10, 1), datetime(2024, 12, 31)),
            ("Consolidation_2024", datetime(2024, 4, 1), datetime(2024, 6, 30)),
            ("Bear_Market_2023", datetime(2023, 1, 1), datetime(2023, 3, 31)),
        ]
        
        results = {}
        
        for period_name, start_date, end_date in periods:
            df = await self.download_period(symbol, start_date, end_date, period_name)
            
            if df is not None:
                results[period_name] = {
                    'start': start_date,
                    'end': end_date,
                    'count': len(df),
                    'first_price': df['close'].iloc[0],
                    'last_price': df['close'].iloc[-1],
                    'return': (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                }
            
            # レート制限を避けるため待機
            await asyncio.sleep(1)
        
        # サマリーを保存
        self._save_summary(results)
        
        return results
    
    def _save_summary(self, results):
        """ダウンロード結果のサマリーを保存"""
        summary = []
        
        for period_name, info in results.items():
            summary.append({
                'period': period_name,
                'start_date': info['start'].strftime('%Y-%m-%d'),
                'end_date': info['end'].strftime('%Y-%m-%d'),
                'data_points': info['count'],
                'first_price': info['first_price'],
                'last_price': info['last_price'],
                'return': info['return']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.data_dir / 'download_summary.csv', index=False)
        
        # レポート作成
        report = f"""
過去データダウンロード完了
================================================================================
ダウンロード日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
ダウンロード期間数: {len(results)}

期間別サマリー:
"""
        
        for _, row in summary_df.iterrows():
            report += f"""
期間: {row['period']}
日付: {row['start_date']} - {row['end_date']}
データ数: {row['data_points']}
初値: ${row['first_price']:,.2f}
終値: ${row['last_price']:,.2f}
リターン: {row['return']:.2%}
"""
        
        with open(self.data_dir / 'download_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary saved to {self.data_dir}")

async def main():
    """メイン実行関数"""
    downloader = HistoricalDataDownloader()
    
    logger.info("Starting historical data download...")
    results = await downloader.download_multiple_periods()
    
    if results:
        logger.info("Download completed successfully!")
        print("\n" + "="*60)
        print("ダウンロード完了")
        print("="*60)
        print(f"ダウンロードした期間数: {len(results)}")
        print(f"保存先: data/historical/")
        print("\n各期間のバックテストを実行する準備ができました。")
    else:
        logger.error("Download failed")

if __name__ == "__main__":
    asyncio.run(main())