#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
現在データ作成スクリプト
最新の期間データを組み合わせてcurrentデータを作成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_current_data():
    """currentデータを作成"""
    data_dir = Path("data/historical")
    
    # 利用可能なデータファイルを確認
    available_files = list(data_dir.glob("BTC_1h_*.csv"))
    logger.info(f"利用可能ファイル: {[f.name for f in available_files]}")
    
    # 最新のデータファイルを使用
    latest_periods = ['2025_07', '2025_06']
    current_data = []
    
    for period in latest_periods:
        file_path = data_dir / f"BTC_1h_{period}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            current_data.append(df)
            logger.info(f"読み込み: {period} - {len(df)}行")
    
    if current_data:
        # データを結合
        combined_df = pd.concat(current_data, axis=0)
        combined_df = combined_df.sort_index()
        
        # 重複除去
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # currentファイルとして保存
        output_path = data_dir / "BTC_1h_current.csv"
        combined_df.to_csv(output_path)
        
        logger.info(f"currentデータ作成完了: {len(combined_df)}行")
        logger.info(f"期間: {combined_df.index.min()} ～ {combined_df.index.max()}")
        logger.info(f"保存先: {output_path}")
        
        return output_path
    else:
        logger.error("currentデータ作成に失敗")
        return None

if __name__ == "__main__":
    create_current_data()