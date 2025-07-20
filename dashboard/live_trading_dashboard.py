#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ライブ取引ダッシュボード
Hyperliquid リアルタイムデータ + AI予測システム
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime
import warnings

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from core.real_data_prediction_system import RealDataPredictionSystem

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="ライブ取引ダッシュボード",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiveTradingDashboard:
    """ライブ取引ダッシュボード"""
    
    def __init__(self):
        self.prediction_system = None
        self.is_initialized = False
        
    @st.cache_resource
    def get_prediction_system():
        """予測システムのシングルトン取得"""
        return RealDataPredictionSystem()
    
    async def initialize_system(self):
        """システム初期化"""
        if not self.is_initialized:
            try:
                with st.spinner("システム初期化中..."):
                    self.prediction_system = self.get_prediction_system()
                    success = await self.prediction_system.initialize()
                    if success:
                        self.is_initialized = True
                        st.success("システム初期化完了")
                        return True
                    else:
                        st.error("システム初期化失敗")
                        return False
            except Exception as e:
                st.error(f"初期化エラー: {e}")
                return False
        return True
    
    def create_price_chart(self, symbol: str, df: pd.DataFrame) -> go.Figure:
        """価格チャート作成"""
        if df.empty:
            return go.Figure().add_annotation(text="データなし", showarrow=False)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # ローソク足
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # 移動平均（データが十分な場合）
        if len(df) >= 20:
            ma_short = df['close'].rolling(7).mean()
            ma_long = df['close'].rolling(20).mean()
            
            fig.add_trace(
                go.Scatter(x=df.index, y=ma_short, name='MA7', 
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=ma_long, name='MA20', 
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # RSI（データが十分な場合）
        if len(df) >= 30:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rsi, name='RSI', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 出来高
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', 
                   marker_color=colors, opacity=0.6),
            row=3, col=1
        )
        
        # レイアウト
        fig.update_layout(
            title=f'{symbol}/USDT ライブチャート',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Y軸
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
    
    def create_prediction_gauge(self, probability: float, symbol: str) -> go.Figure:
        """予測ゲージ作成"""
        color = "green" if probability > 0.6 else "red" if probability < 0.4 else "yellow"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{symbol} 上昇確率"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 70], 'color': "lightgray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, template='plotly_dark')
        return fig
    
    def create_system_status_chart(self, status: dict) -> go.Figure:
        """システム状態チャート作成"""
        symbols = list(status.get('data_cache_status', {}).keys())
        data_counts = [status['data_cache_status'][s]['rows'] for s in symbols]
        
        fig = go.Figure(data=[
            go.Bar(x=symbols, y=data_counts, 
                   marker_color=['green' if count > 100 else 'yellow' if count > 50 else 'red' 
                                for count in data_counts])
        ])
        
        fig.update_layout(
            title="データ取得状況",
            xaxis_title="銘柄",
            yaxis_title="データ数",
            template='plotly_dark',
            height=300
        )
        
        return fig

async def main():
    """メインアプリケーション"""
    st.title("🚀 ライブ取引ダッシュボード")
    st.markdown("### Hyperliquid リアルタイムデータ + AI予測システム")
    
    dashboard = LiveTradingDashboard()
    
    # サイドバー設定
    st.sidebar.header("設定")
    
    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("自動更新", value=True)
    update_interval = st.sidebar.slider("更新間隔 (秒)", 30, 120, 60)
    
    # 銘柄選択
    symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
    selected_symbol = st.sidebar.selectbox("メイン表示銘柄", symbols)
    
    # システム初期化
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    if not st.session_state.system_initialized:
        init_success = await dashboard.initialize_system()
        if init_success:
            st.session_state.system_initialized = True
            st.session_state.dashboard = dashboard
        else:
            st.error("システム初期化に失敗しました")
            return
    else:
        dashboard = st.session_state.dashboard
    
    # メインコンテンツ
    if dashboard.is_initialized:
        # システム状態表示
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # システム状態取得
            status = dashboard.prediction_system.get_system_status()
            predictions = dashboard.prediction_system.get_current_predictions()
            prices = dashboard.prediction_system.get_live_prices()
            
            with col1:
                st.metric("接続状態", "接続中" if status['connected'] else "切断", 
                         delta="正常" if status['connected'] else "エラー")
            
            with col2:
                st.metric("監視銘柄数", status['symbols_monitored'])
            
            with col3:
                st.metric("予測利用可能", status['predictions_available'])
            
            with col4:
                last_update = status.get('last_update')
                if last_update:
                    time_diff = (datetime.now() - last_update).total_seconds()
                    st.metric("最終更新", f"{time_diff:.0f}秒前")
                else:
                    st.metric("最終更新", "未更新")
            
            # メインチャートエリア
            col_chart, col_pred = st.columns([3, 1])
            
            with col_chart:
                st.subheader(f"📈 {selected_symbol} チャート")
                
                # チャートデータ取得
                if selected_symbol in dashboard.prediction_system.data_cache:
                    df = dashboard.prediction_system.data_cache[selected_symbol]
                    chart = dashboard.create_price_chart(selected_symbol, df)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # 価格情報表示
                    if not df.empty:
                        latest_price = df['close'].iloc[-1]
                        prev_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
                        price_change = ((latest_price - prev_price) / prev_price) * 100
                        
                        st.metric(
                            label=f"{selected_symbol} 現在価格",
                            value=f"${latest_price:.2f}",
                            delta=f"{price_change:+.2f}%"
                        )
                else:
                    st.warning(f"{selected_symbol}のデータが取得できていません")
            
            with col_pred:
                st.subheader("🎯 AI予測")
                
                if selected_symbol in predictions:
                    pred = predictions[selected_symbol]
                    
                    # 予測ゲージ
                    gauge = dashboard.create_prediction_gauge(pred['probability'], selected_symbol)
                    st.plotly_chart(gauge, use_container_width=True)
                    
                    # 予測詳細
                    signal_color = "green" if pred['signal'] == 'BUY' else "red" if pred['signal'] == 'SELL' else "gray"
                    st.markdown(f"**シグナル**: :{signal_color}[{pred['signal']}]")
                    
                    st.metric(
                        label="上昇確率",
                        value=f"{pred['probability']:.1%}",
                        delta=f"信頼度: {pred['confidence']:.1%}"
                    )
                    
                    st.caption(f"データ品質: {pred['data_quality']}")
                    
                    if pred['last_update']:
                        update_time = pred['last_update'].strftime('%H:%M:%S')
                        st.caption(f"更新: {update_time}")
                else:
                    st.warning("予測データがありません")
            
            # 全銘柄サマリー
            st.markdown("---")
            st.subheader("📊 全銘柄サマリー")
            
            # 全銘柄予測表
            col_summary, col_status = st.columns(2)
            
            with col_summary:
                summary_data = []
                for symbol in symbols:
                    if symbol in predictions and symbol in prices:
                        pred = predictions[symbol]
                        price = prices[symbol]
                        
                        summary_data.append({
                            '銘柄': symbol,
                            '現在価格': f"${price:.2f}" if price else "N/A",
                            '上昇確率': f"{pred['probability']:.1%}",
                            'シグナル': pred['signal'],
                            '品質': pred['data_quality']
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    st.warning("サマリーデータがありません")
            
            with col_status:
                st.write("**システム状態**")
                
                # データ取得状況チャート
                status_chart = dashboard.create_system_status_chart(status)
                st.plotly_chart(status_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
    
    # 自動更新
    if auto_refresh and dashboard.is_initialized:
        time.sleep(update_interval)
        st.rerun()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"アプリケーションエラー: {e}")