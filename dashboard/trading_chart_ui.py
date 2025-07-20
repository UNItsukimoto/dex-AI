#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取引所風チャートUI
リアルタイム価格チャートとユーザーフレンドリーなインターフェース
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class TradingChartUI:
    """取引所風チャートUI"""
    
    def __init__(self):
        self.api_base_url = "https://api.hyperliquid.xyz"
        
    def create_trading_view_chart(self, symbol: str, timeframe: str = "1h") -> go.Figure:
        """TradingView風のローソク足チャート"""
        try:
            # ローソク足データ取得
            df = self.get_candle_data(symbol, timeframe, 100)
            
            if df.empty:
                # ダミーデータ生成
                df = self.generate_dummy_data(symbol)
            
            # サブプロット作成（価格チャート + ボリューム）
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[f'{symbol}/USD - {timeframe}', 'Volume'],
                row_heights=[0.7, 0.3]
            )
            
            # ローソク足チャート
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol,
                    increasing_line_color='#00D4AA',
                    decreasing_line_color='#FF6B6B'
                ),
                row=1, col=1
            )
            
            # 移動平均線
            if len(df) >= 20:
                df['MA20'] = df['close'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color='#FFD700', width=1)
                    ),
                    row=1, col=1
                )
            
            if len(df) >= 50:
                df['MA50'] = df['close'].rolling(50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA50'],
                        mode='lines',
                        name='MA50',
                        line=dict(color='#FF6347', width=1)
                    ),
                    row=1, col=1
                )
            
            # ボリュームバー
            colors = ['#00D4AA' if close >= open else '#FF6B6B' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # レイアウト設定
            fig.update_layout(
                title=dict(
                    text=f"{symbol}/USD リアルタイムチャート",
                    font=dict(size=20, color='white')
                ),
                template='plotly_dark',
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white')
            )
            
            # X軸設定
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#444444',
                title="時間"
            )
            
            # Y軸設定
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#444444',
                title="価格 (USD)",
                row=1, col=1
            )
            
            fig.update_yaxes(
                title="ボリューム",
                row=2, col=1
            )
            
            return fig
            
        except Exception as e:
            st.error(f"チャート作成エラー: {e}")
            return go.Figure()
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
        """履歴データ取得（統合プラットフォーム用エイリアス）"""
        return self.get_candle_data(symbol, timeframe, count)
    
    def get_candle_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """ローソク足データ取得"""
        try:
            end_time = int(time.time() * 1000)
            
            # 時間枠の設定
            interval_map = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000
            }
            
            interval_ms = interval_map.get(timeframe, 60 * 60 * 1000)
            start_time = end_time - (count * interval_ms)
            
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": timeframe,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            
            response = requests.post(f"{self.api_base_url}/info", json=payload, timeout=10)
            
            if response.status_code == 200:
                candles = response.json()
                
                if not candles:
                    return pd.DataFrame()
                
                df_data = []
                for candle in candles:
                    if isinstance(candle, dict) and 't' in candle:
                        df_data.append({
                            'timestamp': pd.to_datetime(candle['t'], unit='ms'),
                            'open': float(candle['o']),
                            'high': float(candle['h']),
                            'low': float(candle['l']),
                            'close': float(candle['c']),
                            'volume': float(candle['v']) if 'v' in candle else 0.0
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return pd.DataFrame()
    
    def generate_dummy_data(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """ダミーデータ生成（API接続失敗時）"""
        # ベース価格設定
        base_prices = {
            'BTC': 45000,
            'ETH': 3200,
            'SOL': 150,
            'AVAX': 35,
            'DOGE': 0.08,
            'MATIC': 0.75
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # 時系列データ生成
        dates = pd.date_range(end=datetime.now(), periods=count, freq='h')
        
        # ランダムウォーク
        returns = np.random.normal(0, 0.02, count)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # 50%以下には下がらない
        
        # OHLCV作成
        df_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.015))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(100000, 1000000)
            
            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(df_data, index=dates)
        return df
    
    def create_market_overview_widget(self, symbols: List[str]) -> None:
        """マーケット概要ウィジェット"""
        st.subheader("📊 マーケット概要")
        
        # 価格データ取得
        prices = self.get_current_prices(symbols)
        
        # ウィジェット作成
        cols = st.columns(len(symbols))
        
        for i, symbol in enumerate(symbols):
            with cols[i]:
                price = prices.get(symbol, 0)
                
                # 24時間変動率（ダミー）
                change_pct = np.random.uniform(-5, 5)
                delta_color = "normal" if change_pct >= 0 else "inverse"
                
                st.metric(
                    label=symbol,
                    value=f"${price:,.2f}",
                    delta=f"{change_pct:+.2f}%",
                    delta_color=delta_color
                )
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """現在価格取得"""
        try:
            payload = {"type": "allMids"}
            response = requests.post(f"{self.api_base_url}/info", json=payload, timeout=5)
            
            if response.status_code == 200:
                all_mids = response.json()
                prices = {}
                
                for symbol in symbols:
                    if symbol in all_mids:
                        prices[symbol] = float(all_mids[symbol])
                    else:
                        # ダミー価格
                        base_prices = {
                            'BTC': 45000 + np.random.uniform(-1000, 1000),
                            'ETH': 3200 + np.random.uniform(-100, 100),
                            'SOL': 150 + np.random.uniform(-10, 10),
                            'AVAX': 35 + np.random.uniform(-3, 3),
                            'DOGE': 0.08 + np.random.uniform(-0.005, 0.005),
                            'MATIC': 0.75 + np.random.uniform(-0.05, 0.05)
                        }
                        prices[symbol] = base_prices.get(symbol, 1000)
                
                return prices
            else:
                # ダミー価格を返す
                return {symbol: 1000 + np.random.uniform(-100, 100) for symbol in symbols}
                
        except Exception as e:
            print(f"価格取得エラー: {e}")
            return {symbol: 1000 + np.random.uniform(-100, 100) for symbol in symbols}
    
    def create_orderbook_widget(self, symbol: str) -> None:
        """オーダーブック風ウィジェット"""
        st.subheader(f"📋 {symbol} オーダーブック（模擬）")
        
        current_price = self.get_current_prices([symbol])[symbol]
        
        # 模擬オーダーブックデータ
        price_range = np.linspace(current_price * 0.98, current_price * 1.02, 20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**売り注文 (Ask)**")
            asks = []
            for i, price in enumerate(sorted(price_range[10:], reverse=True)):
                size = np.random.uniform(0.1, 5.0)
                asks.append({
                    '価格': f"${price:.2f}",
                    'サイズ': f"{size:.3f}",
                    '累計': f"{sum(np.random.uniform(0.1, 5.0) for _ in range(i+1)):.3f}"
                })
            
            ask_df = pd.DataFrame(asks)
            st.dataframe(ask_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**買い注文 (Bid)**")
            bids = []
            for i, price in enumerate(sorted(price_range[:10], reverse=True)):
                size = np.random.uniform(0.1, 5.0)
                bids.append({
                    '価格': f"${price:.2f}",
                    'サイズ': f"{size:.3f}",
                    '累計': f"{sum(np.random.uniform(0.1, 5.0) for _ in range(i+1)):.3f}"
                })
            
            bid_df = pd.DataFrame(bids)
            st.dataframe(bid_df, use_container_width=True, hide_index=True)
        
        # 現在価格表示
        st.info(f"💰 現在価格: **${current_price:,.2f}**")
    
    def create_trading_interface(self, symbol: str) -> None:
        """取引インターフェース"""
        st.subheader(f"⚡ {symbol} 取引パネル")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🟢 買い注文**")
            buy_amount = st.number_input("金額 (USD)", min_value=10.0, value=100.0, key=f"buy_{symbol}")
            buy_price = st.number_input("価格", value=self.get_current_prices([symbol])[symbol], key=f"buy_price_{symbol}")
            
            if st.button(f"🚀 {symbol} 買い", key=f"buy_btn_{symbol}"):
                st.success(f"✅ {symbol} 買い注文 ${buy_amount:.2f} @ ${buy_price:.2f}")
        
        with col2:
            st.write("**🔴 売り注文**")
            sell_amount = st.number_input("金額 (USD)", min_value=10.0, value=100.0, key=f"sell_{symbol}")
            sell_price = st.number_input("価格", value=self.get_current_prices([symbol])[symbol], key=f"sell_price_{symbol}")
            
            if st.button(f"📉 {symbol} 売り", key=f"sell_btn_{symbol}"):
                st.success(f"✅ {symbol} 売り注文 ${sell_amount:.2f} @ ${sell_price:.2f}")
    
    def create_prediction_overlay(self, fig: go.Figure, prediction: Dict) -> go.Figure:
        """予測オーバーレイ"""
        if not prediction:
            return fig
        
        # 予測ライン追加
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=1)
        
        signal = prediction.get('signal', 'HOLD')
        confidence = prediction.get('confidence', 0)
        current_price = prediction.get('price', 0)
        
        # 予測価格計算（簡易）
        if signal == 'BUY':
            predicted_price = current_price * (1 + confidence * 0.05)
            color = '#00D4AA'
        elif signal == 'SELL':
            predicted_price = current_price * (1 - confidence * 0.05)
            color = '#FF6B6B'
        else:
            predicted_price = current_price
            color = '#FFD700'
        
        # 予測ライン
        fig.add_trace(
            go.Scatter(
                x=[current_time, future_time],
                y=[current_price, predicted_price],
                mode='lines+markers',
                name=f'AI予測 ({confidence:.1%})',
                line=dict(color=color, width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 予測注釈
        fig.add_annotation(
            x=future_time,
            y=predicted_price,
            text=f"{signal}<br>{confidence:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            bgcolor=color,
            bordercolor=color,
            font=dict(color='white')
        )
        
        return fig