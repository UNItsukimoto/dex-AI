#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルタイム予測システム
"""

import sys
from pathlib import Path
import asyncio
import numpy as np
import pandas as pd
import torch
import json
import pickle
from datetime import datetime, timedelta
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import HyperliquidClient
from src.data import HyperliquidDataLoader
from src.features import FeatureManager
from src.models import CryptoGAN
from src.utils import get_logger

logger = get_logger(__name__)

class RealtimePredictor:
    """リアルタイム予測クラス"""
    
    def __init__(self, model_dir: str = 'data/models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_manager = FeatureManager()
        self.client = HyperliquidClient()
        
        # 履歴データ
        self.price_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        
        # 予測バッファ
        self.sequence_buffer = None
        
    def load_model(self):
        """モデルとメタデータの読み込み"""
        # メタデータ
        with open(f"{self.model_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # スケーラー
        with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # モデル
        self.model = CryptoGAN(
            input_size=self.metadata['input_size'],
            sequence_length=self.metadata['sequence_length']
        )
        
        # 最新のチェックポイントを探す
        checkpoint_dir = Path(f"{self.model_dir}/checkpoints")
        checkpoints = list(checkpoint_dir.glob("*_generator.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            self.model.generator.load_checkpoint(str(latest_checkpoint))
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
        else:
            logger.warning("No checkpoint found")
        
        self.model.generator.eval()
        
    async def get_latest_features(self, symbol: str = 'BTC'):
        """最新の特徴量を取得"""
        async with self.client as client:
            # 最新のローソク足データ
            candles = await client.get_candles(symbol, '1h', lookback_days=2)
            
            if not candles:
                return None
            
            # DataFrameに変換
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 不要な列を削除
            for col in ['timestamp_end', 'symbol', 'interval', 'trades']:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            
            # 特徴量を作成
            df_features = self.feature_manager.create_all_features(df)
            
            # 数値列のみ選択（メタデータと同じ列を使用）
            feature_columns = self.metadata['feature_columns']
            available_columns = [col for col in feature_columns if col in df_features.columns]
            df_numeric = df_features[available_columns]
            
            # 欠損値の処理
            df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df_numeric
    
    async def predict_next(self):
        """次の時間の予測"""
        # 最新データの取得
        df_features = await self.get_latest_features()
        
        if df_features is None or len(df_features) < self.metadata['sequence_length']:
            logger.warning("Not enough data for prediction")
            return None
        
        # スケーリング
        features_scaled = self.scaler.transform(df_features.values)
        
        # シーケンスの作成
        sequence = features_scaled[-self.metadata['sequence_length']:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # 予測
        with torch.no_grad():
            prediction = self.model.predict(sequence_tensor, num_predictions=1)
        
        # 逆スケーリング（価格列のインデックスを取得）
        price_idx = self.metadata['feature_columns'].index('close')
        predicted_features = self.scaler.inverse_transform(prediction.numpy())
        predicted_price = predicted_features[0, price_idx]
        
        # 現在価格
        current_price = df_features['close'].iloc[-1]
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_percent': ((predicted_price - current_price) / current_price) * 100,
            'timestamp': datetime.now()
        }
    
    async def run_continuous_prediction(self, interval_seconds: int = 60):
        """継続的な予測実行"""
        logger.info("Starting continuous prediction...")
        
        while True:
            try:
                # 予測
                result = await self.predict_next()
                
                if result:
                    # 履歴に追加
                    self.timestamps.append(result['timestamp'])
                    self.price_history.append(result['current_price'])
                    self.prediction_history.append(result['predicted_price'])
                    
                    # ログ出力
                    logger.info(
                        f"[{result['timestamp'].strftime('%H:%M:%S')}] "
                        f"Current: ${result['current_price']:,.2f} | "
                        f"Predicted: ${result['predicted_price']:,.2f} | "
                        f"Change: {result['change_percent']:+.2f}%"
                    )
                    
                    # アラート条件
                    if abs(result['change_percent']) > 2:
                        logger.warning(
                            f"🚨 ALERT: Large price movement predicted! "
                            f"{result['change_percent']:+.2f}%"
                        )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
            
            # 待機
            await asyncio.sleep(interval_seconds)
    
    def plot_realtime(self):
        """リアルタイムプロット"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def update(frame):
            if len(self.timestamps) < 2:
                return
            
            # 価格プロット
            ax1.clear()
            ax1.plot(self.timestamps, self.price_history, 'b-', label='Actual Price')
            ax1.plot(self.timestamps, self.prediction_history, 'r--', label='Predicted Price')
            ax1.set_ylabel('Price (USD)')
            ax1.set_title('BTC Price - Real-time Prediction')
            ax1.legend()
            ax1.grid(True)
            
            # 誤差プロット
            ax2.clear()
            if len(self.price_history) > 1:
                errors = [
                    (pred - actual) / actual * 100 
                    for pred, actual in zip(self.prediction_history, self.price_history)
                ]
                ax2.plot(self.timestamps, errors, 'g-')
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax2.set_ylabel('Prediction Error (%)')
                ax2.set_xlabel('Time')
                ax2.grid(True)
            
            plt.tight_layout()
        
        ani = FuncAnimation(fig, update, interval=5000, cache_frame_data=False)
        plt.show()

async def main():
    """メイン関数"""
    logger.info("Initializing realtime prediction system...")
    
    # 予測システムの初期化
    predictor = RealtimePredictor()
    predictor.load_model()
    
    # 予測タスクとプロットタスクを並行実行
    prediction_task = asyncio.create_task(
        predictor.run_continuous_prediction(interval_seconds=60)
    )
    
    # 別スレッドでプロット（オプション）
    # import threading
    # plot_thread = threading.Thread(target=predictor.plot_realtime)
    # plot_thread.start()
    
    try:
        await prediction_task
    except KeyboardInterrupt:
        logger.info("Stopping prediction system...")
        prediction_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())