#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取引所スタイルチャートダッシュボード
ローソク足、テクニカル指標、リアルタイム予測を統合
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
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="取引所スタイル仮想通貨チャート",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingChartDashboard:
    """取引所スタイルのチャートダッシュボード"""
    
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
        
    def generate_ohlcv_data(self, symbol, timeframe_minutes, periods=100):
        """OHLCV データを生成（実際の環境ではAPIから取得）"""
        # 基準価格
        base_prices = {
            'BTC/USDT': 67000,
            'ETH/USDT': 3200,
            'SOL/USDT': 180,
            'AVAX/USDT': 30
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # 時間軸作成
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=timeframe_minutes * periods)
        time_range = pd.date_range(start=start_time, end=end_time, periods=periods)
        
        # 価格データ生成
        np.random.seed(hash(symbol) % 1000)
        
        # トレンド + ランダムウォーク
        trend = np.linspace(0, 0.02, periods)  # 2%の上昇トレンド
        returns = np.random.normal(0, 0.01, periods) + trend/periods
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV作成
        data = []
        for i, (timestamp, close) in enumerate(zip(time_range, prices)):
            volatility = abs(np.random.normal(0, 0.005))
            
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # ボリューム（価格変動と逆相関）
            volume = np.random.lognormal(10, 0.5) * (1 + abs(returns[i]) * 10)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low, 
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_technical_indicators(self, df):
        """テクニカル指標を計算"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 移動平均
        df['MA7'] = close.rolling(7).mean()
        df['MA25'] = close.rolling(25).mean()
        df['MA99'] = close.rolling(99).mean()
        
        # ボリンジャーバンド
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_Upper'] = ma20 + (std20 * 2)
        df['BB_Lower'] = ma20 - (std20 * 2)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 出来高平均
        df['Volume_MA'] = volume.rolling(20).mean()
        
        return df
    
    def create_trading_chart(self, df, symbol, show_indicators=True):
        """取引所スタイルのチャートを作成"""
        # サブプロット作成（メインチャート + 指標）
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} 価格チャート', 'ボリューム', 'RSI', 'MACD'),
            row_width=[0.2, 0.1, 0.1, 0.1]
        )
        
        # === メインチャート（ローソク足） ===
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)
        
        # 移動平均線
        if show_indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MA7'],
                name='MA7', line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MA25'],
                name='MA25', line=dict(color='blue', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MA99'],
                name='MA99', line=dict(color='red', width=1)
            ), row=1, col=1)
            
            # ボリンジャーバンド
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['BB_Upper'],
                name='BB上限', line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['BB_Lower'],
                name='BB下限', line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # === ボリュームチャート ===
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='ボリューム',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        if show_indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['Volume_MA'],
                name='ボリューム平均', line=dict(color='blue', width=1),
                showlegend=False
            ), row=2, col=1)
        
        # === RSI ===
        if show_indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['RSI'],
                name='RSI', line=dict(color='purple', width=2),
                showlegend=False
            ), row=3, col=1)
            
            # RSI 買われすぎ/売られすぎライン
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", row=3, col=1)
        
        # === MACD ===
        if show_indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MACD'],
                name='MACD', line=dict(color='blue', width=2),
                showlegend=False
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MACD_Signal'],
                name='Signal', line=dict(color='red', width=2),
                showlegend=False
            ), row=4, col=1)
            
            # MACDヒストグラム
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=df['timestamp'],
                y=df['MACD_Histogram'],
                name='ヒストグラム',
                marker_color=colors,
                showlegend=False
            ), row=4, col=1)
        
        # レイアウト設定
        fig.update_layout(
            title=f'{symbol} - 取引所スタイルチャート',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(color='white')
        )
        
        # X軸の設定
        fig.update_xaxes(
            type='date',
            showgrid=True,
            gridwidth=1,
            gridcolor='#444444'
        )
        
        # Y軸の設定
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#444444'
        )
        
        return fig
    
    def get_prediction_for_symbol(self, symbol):
        """シンボルの予測を取得"""
        # シミュレートされた予測データ
        np.random.seed(hash(symbol) % 1000 + int(time.time() / 60))
        
        prediction = np.random.normal(0, 0.02)
        probability = 1 / (1 + np.exp(-prediction * 10)) # シグモイド関数
        confidence = np.random.uniform(0.6, 0.95)
        
        return {
            'symbol': symbol,
            'predicted_return': prediction,
            'up_probability': probability,
            'confidence': confidence,
            'signal': 'BUY' if probability > 0.6 else 'SELL' if probability < 0.4 else 'HOLD'
        }

def create_market_overview(dashboard):
    """マーケット概要を作成"""
    data = []
    for symbol in dashboard.symbols:
        df = dashboard.generate_ohlcv_data(symbol, 60, 24)  # 24時間データ
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-25] if len(df) > 24 else df['close'].iloc[0]
        change_24h = (current_price - prev_price) / prev_price
        
        prediction = dashboard.get_prediction_for_symbol(symbol)
        
        data.append({
            'シンボル': symbol,
            '価格': f"${current_price:,.2f}",
            '24h変動': f"{change_24h:+.2%}",
            'AI予測': f"{prediction['up_probability']:.1%}",
            'シグナル': prediction['signal'],
            '信頼度': f"{prediction['confidence']:.1%}"
        })
    
    return pd.DataFrame(data)

def create_prediction_panel(symbol, dashboard):
    """予測パネルを作成"""
    prediction = dashboard.get_prediction_for_symbol(symbol)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "上昇確率",
            f"{prediction['up_probability']:.1%}",
            f"{prediction['predicted_return']:+.2%}"
        )
    
    with col2:
        signal_color = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '🟡'
        }
        st.metric(
            "AIシグナル",
            f"{signal_color[prediction['signal']]} {prediction['signal']}"
        )
    
    with col3:
        st.metric(
            "信頼度",
            f"{prediction['confidence']:.1%}"
        )

def main():
    """メインダッシュボード"""
    st.title("📈 取引所スタイル仮想通貨チャート")
    
    # ヘッダー
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("🔄 **リアルタイム更新中**")
    with col2:
        if st.button("🔄 更新", type="secondary"):
            st.rerun()
    with col3:
        st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    # ダッシュボードインスタンス
    dashboard = TradingChartDashboard()
    
    # サイドバー設定
    st.sidebar.header("📊 チャート設定")
    
    selected_symbol = st.sidebar.selectbox(
        "通貨ペア",
        dashboard.symbols,
        index=0
    )
    
    selected_timeframe = st.sidebar.selectbox(
        "時間足",
        list(dashboard.timeframes.keys()),
        index=3  # 1時間をデフォルト
    )
    
    show_indicators = st.sidebar.checkbox("テクニカル指標を表示", value=True)
    show_predictions = st.sidebar.checkbox("AI予測を表示", value=True)
    
    periods = st.sidebar.slider("表示期間", 50, 200, 100)
    
    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("自動更新", value=False)
    if auto_refresh:
        refresh_rate = st.sidebar.selectbox("更新間隔", [30, 60, 120], format_func=lambda x: f"{x}秒")
    
    # マーケット概要
    st.subheader("📊 マーケット概要")
    market_overview = create_market_overview(dashboard)
    st.dataframe(market_overview, use_container_width=True)
    
    # メインチャート
    st.subheader(f"📈 {selected_symbol} チャート")
    
    # AI予測パネル
    if show_predictions:
        st.subheader("🤖 AI予測")
        create_prediction_panel(selected_symbol, dashboard)
        st.markdown("---")
    
    # チャートデータ生成
    timeframe_minutes = dashboard.timeframes[selected_timeframe]
    df = dashboard.generate_ohlcv_data(selected_symbol, timeframe_minutes, periods)
    df = dashboard.calculate_technical_indicators(df)
    
    # チャート作成・表示
    try:
        chart = dashboard.create_trading_chart(df, selected_symbol, show_indicators)
        st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"チャート表示エラー: {str(e)}")
        st.info("シンプルな価格チャートを表示します")
        
        # フォールバックチャート
        simple_chart = go.Figure()
        simple_chart.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='価格',
            line=dict(color='blue', width=2)
        ))
        simple_chart.update_layout(
            title=f'{selected_symbol} 価格推移',
            xaxis_title='時間',
            yaxis_title='価格 (USDT)',
            height=400
        )
        st.plotly_chart(simple_chart, use_container_width=True)
    
    # 価格統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("現在価格", f"${df['close'].iloc[-1]:,.2f}")
    
    with col2:
        change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        st.metric("変動", f"{change:+.2%}")
    
    with col3:
        high_24h = df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()
        st.metric("24h高値", f"${high_24h:,.2f}")
    
    with col4:
        low_24h = df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()
        st.metric("24h安値", f"${low_24h:,.2f}")
    
    # 自動更新処理
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    📈 取引所スタイルチャート | AIによる予測は教育目的のみ | 実際の投資判断にはご注意ください
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()