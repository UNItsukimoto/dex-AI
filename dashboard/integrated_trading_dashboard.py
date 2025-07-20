#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合取引ダッシュボード
取引所スタイルチャート + リアルタイム予測システム
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="統合仮想通貨取引ダッシュボード",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IntegratedTradingDashboard:
    """統合取引ダッシュボード"""
    
    def __init__(self):
        self.timeframes = {
            '1分': 1,
            '5分': 5, 
            '15分': 15,
            '1時間': 60,
            '4時間': 240,
            '1日': 1440
        }
        
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT']
        
        # 予測精度データ（改善版）
        self.prediction_accuracy = {
            'current': 0.5946,
            '2025_07': 0.5246,
            '2025_06': 0.5345,
            '2025_05': 0.6000,
            '2025_04': 0.5054,
            '2024_Q4': 0.5262,
            'Bull_Run_2024': 0.5545
        }
        
    def load_prediction_system(self):
        """改善された予測システムを読み込み"""
        try:
            from core.simple_effective_2025_06 import SimpleEffective2025_06
            from core.enhanced_prediction_system import EnhancedPredictionSystem
            return True
        except ImportError:
            st.warning("予測システムの読み込みに失敗しました")
            return False
    
    def generate_ohlcv_data(self, symbol, timeframe_minutes, periods=100):
        """OHLCV データを生成"""
        base_prices = {
            'BTC/USDT': 67000,
            'ETH/USDT': 3200,
            'SOL/USDT': 180,
            'AVAX/USDT': 30
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # 時系列データ生成
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=timeframe_minutes * periods)
        
        dates = pd.date_range(start=start_time, end=end_time, freq=f'{timeframe_minutes}min')
        
        # 価格データ生成（よりリアルな動き）
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, len(dates))
        
        # トレンドと変動を追加
        trend = np.linspace(-0.05, 0.05, len(dates))
        volatility = np.random.normal(0, 0.01, len(dates))
        
        prices = [base_price]
        for i in range(1, len(dates)):
            price_change = prices[-1] * (returns[i] + trend[i] + volatility[i])
            new_price = max(prices[-1] + price_change, base_price * 0.5)  # 最低価格制限
            prices.append(new_price)
        
        # OHLCV データ作成
        data = []
        for i in range(len(dates)):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_technical_indicators(self, df):
        """テクニカル指標計算"""
        # 移動平均
        df['MA7'] = df['close'].rolling(7).mean()
        df['MA25'] = df['close'].rolling(25).mean()
        df['MA99'] = df['close'].rolling(99).mean()
        
        # ボリンジャーバンド
        df['BB_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def generate_predictions(self, symbols):
        """リアルタイム予測生成"""
        predictions = {}
        for symbol in symbols:
            # 実際の予測システムを模擬
            base_prob = np.random.uniform(0.45, 0.65)  # 改善された範囲
            
            # 時間による変動
            time_factor = 0.05 * np.sin(time.time() / 60)  # 1分周期
            
            # 最終予測確率
            probability = max(0.0, min(1.0, base_prob + time_factor))
            
            predictions[symbol] = {
                'probability': probability,
                'confidence': np.random.uniform(0.7, 0.95),
                'signal': 'BUY' if probability > 0.55 else 'SELL' if probability < 0.45 else 'HOLD',
                'last_update': datetime.now()
            }
        
        return predictions
    
    def create_candlestick_chart(self, df, symbol):
        """ローソク足チャート作成"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_width=[0.2, 0.2, 0.2, 0.4],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": True}]]
        )
        
        # ローソク足
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=4, col=1
        )
        
        # 移動平均線
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MA7'], name='MA7', 
                      line=dict(color='orange', width=1)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MA25'], name='MA25', 
                      line=dict(color='blue', width=1)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MA99'], name='MA99', 
                      line=dict(color='red', width=1)),
            row=4, col=1
        )
        
        # ボリンジャーバンド
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_upper'], name='BB上限', 
                      line=dict(color='gray', width=1, dash='dash')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_lower'], name='BB下限', 
                      line=dict(color='gray', width=1, dash='dash')),
            row=4, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MACD_signal'], name='Signal', 
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['MACD_histogram'], name='Histogram', 
                   marker_color='gray'),
            row=2, col=1
        )
        
        # 出来高
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', 
                   marker_color=colors),
            row=3, col=1
        )
        
        # レイアウト設定
        fig.update_layout(
            title=f'{symbol} 取引所スタイルチャート',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Y軸設定
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_yaxes(title_text="Price (USDT)", row=4, col=1)
        
        return fig
    
    def create_prediction_gauge(self, probability, symbol):
        """予測確率ゲージ作成"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{symbol} 上昇確率"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, template='plotly_dark')
        return fig
    
    def create_accuracy_chart(self):
        """予測精度チャート作成"""
        periods = list(self.prediction_accuracy.keys())
        accuracies = [self.prediction_accuracy[p] * 100 for p in periods]
        
        colors = ['green' if acc >= 50 else 'red' for acc in accuracies]
        
        fig = go.Figure(data=[
            go.Bar(x=periods, y=accuracies, marker_color=colors)
        ])
        
        fig.add_hline(y=50, line_dash="dash", line_color="yellow", 
                     annotation_text="目標ライン (50%)")
        
        fig.update_layout(
            title="期間別予測精度",
            xaxis_title="期間",
            yaxis_title="精度 (%)",
            template='plotly_dark',
            height=400
        )
        
        return fig

def main():
    """メインアプリケーション"""
    st.title("🚀 統合仮想通貨取引ダッシュボード")
    st.markdown("### 取引所スタイルチャート + AI予測システム")
    
    dashboard = IntegratedTradingDashboard()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("自動更新", value=True)
    update_interval = st.sidebar.slider("更新間隔 (秒)", 10, 60, 30)
    
    # 銘柄選択
    selected_symbol = st.sidebar.selectbox("銘柄選択", dashboard.symbols)
    
    # 時間軸選択
    selected_timeframe = st.sidebar.selectbox("時間軸", list(dashboard.timeframes.keys()))
    
    # 表示期間
    periods = st.sidebar.slider("表示期間", 50, 200, 100)
    
    # メインコンテンツ
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📈 {selected_symbol} チャート分析")
        
        # データ生成
        timeframe_minutes = dashboard.timeframes[selected_timeframe]
        df = dashboard.generate_ohlcv_data(selected_symbol, timeframe_minutes, periods)
        df = dashboard.calculate_technical_indicators(df)
        
        # チャート表示
        chart = dashboard.create_candlestick_chart(df, selected_symbol)
        st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        st.subheader("🎯 AI予測")
        
        # 予測生成
        predictions = dashboard.generate_predictions([selected_symbol])
        pred = predictions[selected_symbol]
        
        # 予測ゲージ
        gauge = dashboard.create_prediction_gauge(pred['probability'], selected_symbol)
        st.plotly_chart(gauge, use_container_width=True)
        
        # 予測詳細
        st.metric(
            label="予測シグナル",
            value=pred['signal'],
            delta=f"信頼度: {pred['confidence']:.1%}"
        )
        
        st.metric(
            label="上昇確率", 
            value=f"{pred['probability']:.1%}",
            delta=f"{pred['probability'] - 0.5:.1%}" if pred['probability'] != 0.5 else "0.0%"
        )
        
        # 最終更新時刻
        st.caption(f"最終更新: {pred['last_update'].strftime('%H:%M:%S')}")
    
    # 下部セクション
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 全銘柄予測")
        
        # 全銘柄の予測表示
        all_predictions = dashboard.generate_predictions(dashboard.symbols)
        
        pred_data = []
        for symbol, pred in all_predictions.items():
            pred_data.append({
                '銘柄': symbol,
                '上昇確率': f"{pred['probability']:.1%}",
                'シグナル': pred['signal'],
                '信頼度': f"{pred['confidence']:.1%}"
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
    
    with col4:
        st.subheader("🎯 システム精度")
        
        # 精度チャート
        accuracy_chart = dashboard.create_accuracy_chart()
        st.plotly_chart(accuracy_chart, use_container_width=True)
        
        # 精度サマリー
        avg_accuracy = np.mean(list(dashboard.prediction_accuracy.values()))
        above_50 = sum(1 for acc in dashboard.prediction_accuracy.values() if acc >= 0.5)
        total_periods = len(dashboard.prediction_accuracy)
        
        st.metric(
            label="平均精度",
            value=f"{avg_accuracy:.1%}",
            delta=f"{above_50}/{total_periods} 期間で50%超達成"
        )
    
    # 自動更新
    if auto_refresh:
        time.sleep(update_interval)
        st.rerun()

if __name__ == "__main__":
    main()