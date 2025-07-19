#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルタイム仮想通貨予測ダッシュボード
自動更新機能付きWebUI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import warnings
warnings.filterwarnings('ignore')

# ダッシュボード設定
st.set_page_config(
    page_title="リアルタイム仮想通貨AI予測",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealtimeCryptoDashboard:
    """リアルタイム仮想通貨予測ダッシュボード"""
    
    def __init__(self):
        self.major_coins = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum', 
            'SOL': 'Solana',
            'AVAX': 'Avalanche',
            'NEAR': 'NEAR Protocol',
            'ARB': 'Arbitrum',
            'OP': 'Optimism',
            'MATIC': 'Polygon'
        }
        
        # リアルタイム用の基本価格（実際の環境では API から取得）
        self.base_prices = {
            'BTC': 118259.0,
            'ETH': 3150.0,
            'SOL': 185.0,
            'AVAX': 28.5,
            'NEAR': 5.8,
            'ARB': 0.85,
            'OP': 1.95,
            'MATIC': 0.48
        }
        
    def get_current_price(self, symbol):
        """現在価格を取得（シミュレート + 変動）"""
        base = self.base_prices.get(symbol, 100.0)
        # 時間ベースで価格を微変動させる
        time_factor = time.time() % 3600  # 1時間サイクル
        volatility = 0.005  # 0.5%の変動
        price_change = np.sin(time_factor / 600) * volatility  # 10分周期
        return base * (1 + price_change)
        
    def simulate_realtime_prediction(self, symbol):
        """リアルタイム予測をシミュレート"""
        # 時間ベースのシード
        current_minute = int(time.time() / 60)
        np.random.seed((hash(symbol) + current_minute) % 10000)
        
        # 10個のモデル予測値（時間で変動）
        base_trend = np.sin(time.time() / 1800) * 0.01  # 30分周期のトレンド
        model_predictions = np.random.normal(base_trend, 0.015, 10)
        
        model_names = ['RF', 'ExtraTrees', 'GradientBoost', 'XGBoost', 'LightGBM', 
                      'Ridge', 'Lasso', 'ElasticNet', 'SVR_RBF', 'SVR_Linear']
        
        # 上昇確率計算
        up_probability = (model_predictions > 0).mean()
        
        # 平均予測リターン
        avg_return = model_predictions.mean()
        
        # 信頼度（予測の一致度）
        confidence = 1 - np.std(model_predictions) / 0.02
        confidence = max(0, min(1, confidence))
        
        # 現在価格
        current_price = self.get_current_price(symbol)
        
        # 価格変動（前回との差）
        prev_price = self.base_prices.get(symbol, 100.0)
        price_change = (current_price - prev_price) / prev_price
        
        return {
            'symbol': symbol,
            'name': self.major_coins[symbol],
            'current_price': current_price,
            'price_change_24h': price_change,
            'predicted_return': avg_return,
            'up_probability': up_probability,
            'confidence': confidence,
            'model_predictions': dict(zip(model_names, model_predictions)),
            'consensus': self._get_consensus_strength(up_probability),
            'last_update': datetime.now()
        }
    
    def _get_consensus_strength(self, probability):
        """合意度強度を取得"""
        if abs(probability - 0.5) > 0.3:
            return 'Strong'
        elif abs(probability - 0.5) > 0.15:
            return 'Moderate'
        else:
            return 'Weak'

def create_live_price_chart(predictions_data):
    """ライブ価格チャートを作成"""
    symbols = list(predictions_data.keys())
    prices = [predictions_data[s]['current_price'] for s in symbols]
    changes = [predictions_data[s]['price_change_24h'] * 100 for s in symbols]
    
    colors = ['green' if c >= 0 else 'red' for c in changes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=prices,
            marker_color=colors,
            text=[f"${p:,.0f}<br>{c:+.2f}%" for p, c in zip(prices, changes)],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="リアルタイム価格",
        xaxis_title="銘柄",
        yaxis_title="価格 (USD)",
        height=300,
        showlegend=False
    )
    
    return fig

def create_probability_timeline(predictions_data, symbol):
    """確率の時系列チャート（シミュレート）"""
    # 過去1時間のデータをシミュレート
    times = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -5)]
    
    # 確率の変化をシミュレート
    base_prob = predictions_data[symbol]['up_probability']
    probabilities = []
    
    for i, t in enumerate(times):
        noise = np.sin(i * 0.3) * 0.1 + np.random.normal(0, 0.05)
        prob = max(0, min(1, base_prob + noise))
        probabilities.append(prob * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=probabilities,
        mode='lines+markers',
        name=f'{symbol} 上昇確率',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% (中立)")
    
    fig.update_layout(
        title=f"{symbol} 上昇確率の推移 (過去1時間)",
        xaxis_title="時刻",
        yaxis_title="上昇確率 (%)",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig

def create_realtime_gauge(probability, symbol, last_update):
    """リアルタイム更新ゲージ"""
    try:
        probability = max(0.0, min(1.0, float(probability)))
        
        # 色の決定
        if probability >= 0.7:
            color = "green"
        elif probability >= 0.5:
            color = "orange" 
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {
                'text': f"{symbol}<br>上昇確率<br><span style='font-size:10px;'>{last_update.strftime('%H:%M:%S')}</span>", 
                'font': {'size': 12}
            },
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"}, 
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "gray", 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            number = {'suffix': "%", 'font': {'size': 14}}
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=15, r=15, t=60, b=15),
            font=dict(family="Arial", size=10)
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error: {str(e)[:30]}...",
            x=0.5, y=0.5,
            showarrow=False
        )

def main():
    """メインダッシュボード"""
    st.title("⚡ リアルタイム仮想通貨AI予測ダッシュボード")
    
    # ヘッダー情報
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("🔄 **自動更新中** - 30秒ごとに予測データが更新されます")
    with col2:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"🕐 {current_time}")
    with col3:
        if st.button("🔄 手動更新", type="secondary"):
            st.rerun()
    
    st.markdown("---")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # 更新間隔設定
    update_interval = st.sidebar.selectbox(
        "自動更新間隔",
        options=[30, 60, 120, 300],
        format_func=lambda x: f"{x}秒",
        index=0
    )
    
    # 銘柄選択
    all_coins = ['BTC', 'ETH', 'SOL', 'AVAX', 'NEAR', 'ARB', 'OP', 'MATIC']
    selected_coins = st.sidebar.multiselect(
        "表示する銘柄:",
        options=all_coins,
        default=['BTC', 'ETH', 'SOL', 'AVAX']
    )
    
    if not selected_coins:
        st.warning("⚠️ 少なくとも1つの銘柄を選択してください")
        return
    
    # 表示オプション
    show_charts = st.sidebar.checkbox("価格チャートを表示", value=True)
    show_timeline = st.sidebar.checkbox("確率推移を表示", value=True) 
    auto_refresh = st.sidebar.checkbox("自動更新", value=True)
    
    # ダッシュボードインスタンス
    dashboard = RealtimeCryptoDashboard()
    
    # 自動更新のプレースホルダー
    if auto_refresh:
        placeholder = st.empty()
        
        while True:
            with placeholder.container():
                # 現在の予測データ取得
                predictions = {}
                for coin in selected_coins:
                    predictions[coin] = dashboard.simulate_realtime_prediction(coin)
                
                # ライブ統計表示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_prob = np.mean([p['up_probability'] for p in predictions.values()])
                    st.metric("平均上昇確率", f"{avg_prob:.1%}")
                
                with col2:
                    bullish_count = sum(1 for p in predictions.values() if p['up_probability'] > 0.6)
                    st.metric("強気銘柄", f"{bullish_count}/{len(selected_coins)}")
                
                with col3:
                    high_conf = sum(1 for p in predictions.values() if p['confidence'] > 0.7)
                    st.metric("高信頼度", f"{high_conf}/{len(selected_coins)}")
                
                with col4:
                    avg_return = np.mean([p['predicted_return'] for p in predictions.values()])
                    st.metric("平均予測リターン", f"{avg_return:.2%}")
                
                # ライブ価格チャート
                if show_charts:
                    st.subheader("📈 ライブ価格")
                    price_chart = create_live_price_chart(predictions)
                    st.plotly_chart(price_chart, use_container_width=True)
                
                # リアルタイム確率ゲージ
                st.subheader("🎯 リアルタイム上昇確率")
                
                # 4列ずつ表示
                for i in range(0, len(selected_coins), 4):
                    batch = selected_coins[i:i+4]
                    cols = st.columns(len(batch))
                    
                    for j, coin in enumerate(batch):
                        with cols[j]:
                            data = predictions[coin]
                            gauge_fig = create_realtime_gauge(
                                data['up_probability'], 
                                coin, 
                                data['last_update']
                            )
                            st.plotly_chart(gauge_fig, use_container_width=True)
                
                # ライブデータテーブル
                st.subheader("📊 ライブ予測データ")
                
                live_data = []
                for symbol, data in predictions.items():
                    live_data.append({
                        '銘柄': symbol,
                        '現在価格': f"${data['current_price']:,.2f}",
                        '24h変動': f"{data['price_change_24h']:+.2%}",
                        '上昇確率': f"{data['up_probability']:.1%}",
                        '予測リターン': f"{data['predicted_return']:+.2%}",
                        '信頼度': f"{data['confidence']:.1%}",
                        '合意度': data['consensus'],
                        '更新時刻': data['last_update'].strftime('%H:%M:%S')
                    })
                
                df = pd.DataFrame(live_data)
                st.dataframe(df, use_container_width=True)
                
                # 確率推移チャート
                if show_timeline and len(selected_coins) <= 4:
                    st.subheader("📈 確率推移")
                    
                    cols = st.columns(len(selected_coins))
                    for i, coin in enumerate(selected_coins):
                        with cols[i]:
                            timeline_fig = create_probability_timeline(predictions, coin)
                            st.plotly_chart(timeline_fig, use_container_width=True)
                
                # アラート表示
                st.subheader("🚨 アラート")
                
                alerts = []
                for symbol, data in predictions.items():
                    if data['up_probability'] >= 0.8:
                        alerts.append(f"🟢 {symbol}: 強い上昇シグナル ({data['up_probability']:.1%})")
                    elif data['up_probability'] <= 0.2:
                        alerts.append(f"🔴 {symbol}: 強い下降シグナル ({data['up_probability']:.1%})")
                    elif data['confidence'] >= 0.9:
                        alerts.append(f"⭐ {symbol}: 高信頼度予測 ({data['confidence']:.1%})")
                
                if alerts:
                    for alert in alerts:
                        st.info(alert)
                else:
                    st.info("現在、アラート対象の銘柄はありません")
                
                # フッター
                st.markdown("---")
                st.markdown(f"""
                <div style='text-align: center; color: #666; font-size: 0.9em;'>
                ⚡ 自動更新: {update_interval}秒間隔 | 
                最終更新: {datetime.now().strftime('%H:%M:%S')} | 
                表示銘柄: {len(selected_coins)}
                </div>
                """, unsafe_allow_html=True)
            
            # 指定間隔で更新
            time.sleep(update_interval)
            st.rerun()
    
    else:
        # 手動更新モード
        st.info("🔄 手動更新モードです。「手動更新」ボタンを押して最新データを取得してください。")
        
        predictions = {}
        for coin in selected_coins:
            predictions[coin] = dashboard.simulate_realtime_prediction(coin)
        
        # 静的表示
        live_data = []
        for symbol, data in predictions.items():
            live_data.append({
                '銘柄': symbol,
                '現在価格': f"${data['current_price']:,.2f}",
                '上昇確率': f"{data['up_probability']:.1%}",
                '予測リターン': f"{data['predicted_return']:+.2%}",
                '信頼度': f"{data['confidence']:.1%}",
                '更新時刻': data['last_update'].strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(live_data)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()