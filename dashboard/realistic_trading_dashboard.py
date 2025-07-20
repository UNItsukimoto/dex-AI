#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルな体験のトレーディングダッシュボード
実際の取引により近い体験を提供
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from core.realistic_paper_trading import RealisticPaperTradingEngine, OrderSide, OrderType
from core.enhanced_ai_trader import EnhancedAITrader

# ページ設定
st.set_page_config(
    page_title="🚀 Realistic Trading Experience",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（リアルな取引所風）
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00D4AA, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .trading-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    .profit-positive {
        color: #00D4AA;
        font-weight: bold;
    }
    .profit-negative {
        color: #FF6B6B;
        font-weight: bold;
    }
    .price-up {
        color: #00D4AA;
    }
    .price-down {
        color: #FF6B6B;
    }
    .order-buy {
        background: linear-gradient(145deg, #00D4AA, #00B894);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
    }
    .order-sell {
        background: linear-gradient(145deg, #FF6B6B, #E55656);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #00D4AA;
    }
    .status-warning {
        background-color: #FFD700;
    }
    .status-offline {
        background-color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

def init_realistic_trader():
    """リアルなトレーダー初期化"""
    if 'realistic_trader' not in st.session_state:
        st.session_state.realistic_trader = RealisticPaperTradingEngine(10000.0)
        st.session_state.ai_trader = EnhancedAITrader(10000.0)
        st.session_state.last_update = time.time()
        st.session_state.total_trades = 0
    return st.session_state.realistic_trader, st.session_state.ai_trader

def create_price_chart(symbol: str, trader) -> go.Figure:
    """リアルタイム価格チャート"""
    price_history = trader.get_price_history(symbol, 24)
    
    if not price_history:
        # ダミーデータ生成
        now = datetime.now()
        times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
        current_price = trader.live_prices.get(symbol, 1000)
        prices = [current_price * (1 + np.random.normal(0, 0.02)) for _ in times]
        price_history = [{'timestamp': t, 'price': p} for t, p in zip(times, prices)]
    
    df = pd.DataFrame(price_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # 価格ライン
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#00D4AA', width=2)
    ))
    
    # 現在価格を強調
    if len(df) > 0:
        current_price = df['price'].iloc[-1]
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#FFD700",
            annotation_text=f"${current_price:,.2f}"
        )
    
    fig.update_layout(
        title=f"{symbol}/USD リアルタイム価格",
        template='plotly_dark',
        height=400,
        showlegend=False,
        xaxis_title="時間",
        yaxis_title="価格 (USD)"
    )
    
    return fig

def main():
    # ヘッダー
    st.markdown('<h1 class="main-header">🚀 Realistic Trading Experience</h1>', unsafe_allow_html=True)
    st.markdown("**実際の取引により近いペーパートレード体験**")
    
    # トレーダー初期化
    trader, ai_trader = init_realistic_trader()
    
    # 自動更新（5秒ごと）
    if time.time() - st.session_state.last_update > 5:
        trader.update_live_prices()
        st.session_state.last_update = time.time()
        st.rerun()
    
    # サイドバー
    with st.sidebar:
        st.header("🎛️ Trading Control")
        
        # ライブステータス
        st.markdown('<div class="status-indicator status-online"></div>Live Market Data', unsafe_allow_html=True)
        st.write(f"最終更新: {datetime.now().strftime('%H:%M:%S')}")
        
        # アカウント情報
        account = trader.get_account_summary()
        
        st.subheader("💰 Account")
        
        balance_color = "profit-positive" if account['daily_pnl'] >= 0 else "profit-negative"
        
        st.markdown(f"""
        <div class="trading-card">
            <div>残高: <strong>${account['balance']:,.2f}</strong></div>
            <div>エクイティ: <strong>${account['equity']:,.2f}</strong></div>
            <div class="{balance_color}">日次損益: ${account['daily_pnl']:,.2f} ({account['daily_pnl_pct']:.2%})</div>
            <div>証拠金率: {account['margin_ratio']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # マーケットステータス
        st.subheader("📊 Market Status")
        market_summary = trader.get_market_summary()
        
        for symbol, data in list(market_summary.items())[:4]:
            change_color = "price-up" if data['change_24h'] >= 0 else "price-down"
            change_symbol = "↗" if data['change_24h'] >= 0 else "↘"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                <span><strong>{symbol}</strong></span>
                <span class="{change_color}">${data['price']:,.2f} {change_symbol}{abs(data['change_24h']):.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        # クイックアクション
        st.subheader("⚡ Quick Actions")
        
        if st.button("🎯 AI Prediction", use_container_width=True, key="ai_predict"):
            with st.spinner("AI analyzing market..."):
                try:
                    ai_trader.execute_enhanced_strategy()
                    st.success("✅ AI Prediction Complete!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_data"):
            trader.update_live_prices()
            st.rerun()
    
    # メインエリア
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trading", "📊 Portfolio", "📋 Orders & History", "🤖 AI Insights"])
    
    with tab1:
        # トレーディングタブ
        col_chart, col_trading = st.columns([2, 1])
        
        with col_chart:
            # 銘柄選択とチャート
            symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE', 'MATIC']
            selected_symbol = st.selectbox("Select Asset", symbols, key="chart_symbol")
            
            # リアルタイムチャート
            chart = create_price_chart(selected_symbol, trader)
            st.plotly_chart(chart, use_container_width=True)
            
            # 価格情報
            prices = trader.get_live_prices()
            market_data = trader.get_market_summary().get(selected_symbol, {})
            
            col_price1, col_price2, col_price3, col_price4 = st.columns(4)
            
            with col_price1:
                st.metric("Current Price", f"${prices.get(selected_symbol, 0):,.2f}")
            
            with col_price2:
                change_24h = market_data.get('change_24h', 0)
                st.metric("24h Change", f"{change_24h:+.2f}%")
            
            with col_price3:
                st.metric("24h High", f"${market_data.get('day_high', 0):,.2f}")
            
            with col_price4:
                st.metric("24h Low", f"${market_data.get('day_low', 0):,.2f}")
        
        with col_trading:
            # 注文パネル
            st.subheader("📝 Place Order")
            
            # 注文タイプ
            order_type = st.radio("Order Type", ["Market", "Limit"], horizontal=True)
            
            # 売買選択
            trade_side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
            
            # 数量入力
            current_price = prices.get(selected_symbol, 0)
            
            if order_type == "Market":
                amount_usd = st.number_input("Amount (USD)", min_value=10.0, value=500.0, step=10.0)
                quantity = amount_usd / current_price if current_price > 0 else 0
                st.write(f"Quantity: {quantity:.6f} {selected_symbol}")
                price_input = None
            else:
                quantity = st.number_input(f"Quantity ({selected_symbol})", min_value=0.0001, value=0.1, step=0.0001, format="%.6f")
                price_input = st.number_input("Price (USD)", min_value=1.0, value=current_price, step=1.0)
                amount_usd = quantity * price_input
            
            # 注文概算
            fee = amount_usd * 0.001  # 0.1% fee
            total = amount_usd + fee if trade_side == "Buy" else amount_usd - fee
            
            st.markdown(f"""
            **Order Summary:**
            - Amount: ${amount_usd:.2f}
            - Fee: ${fee:.2f}
            - Total: ${total:.2f}
            """)
            
            # 注文実行
            side = OrderSide.BUY if trade_side == "Buy" else OrderSide.SELL
            otype = OrderType.MARKET if order_type == "Market" else OrderType.LIMIT
            
            button_class = "order-buy" if trade_side == "Buy" else "order-sell"
            
            if st.button(f"🚀 {trade_side} {selected_symbol}", 
                        type="primary" if trade_side == "Buy" else "secondary",
                        use_container_width=True,
                        key="place_order"):
                try:
                    order_id = trader.place_order(
                        symbol=selected_symbol,
                        side=side,
                        order_type=otype,
                        quantity=quantity,
                        price=price_input
                    )
                    
                    if order_id:
                        st.success(f"✅ Order placed successfully!")
                        st.write(f"Order ID: {order_id}")
                        st.session_state.total_trades += 1
                        
                        if order_type == "Market":
                            st.balloons()
                        
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Order failed. Check your balance and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Order error: {e}")
            
            # 現在のポジション
            st.subheader("📍 Current Position")
            positions = trader.get_positions()
            
            if selected_symbol in positions:
                pos = positions[selected_symbol]
                pnl_color = "profit-positive" if pos['unrealized_pnl'] >= 0 else "profit-negative"
                
                st.markdown(f"""
                <div class="trading-card">
                    <div>Quantity: <strong>{pos['quantity']:.6f}</strong></div>
                    <div>Entry Price: <strong>${pos['entry_price']:,.2f}</strong></div>
                    <div>Current Price: <strong>${pos['current_price']:,.2f}</strong></div>
                    <div class="{pnl_color}">P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:+.2f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                
                # ポジション決済ボタン
                if st.button(f"Close {selected_symbol} Position", use_container_width=True, key="close_position"):
                    try:
                        close_side = OrderSide.SELL if pos['quantity'] > 0 else OrderSide.BUY
                        close_order = trader.place_order(
                            symbol=selected_symbol,
                            side=close_side,
                            order_type=OrderType.MARKET,
                            quantity=abs(pos['quantity'])
                        )
                        
                        if close_order:
                            st.success("✅ Position closed!")
                            time.sleep(1)
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Close error: {e}")
            else:
                st.info("No position in this asset")
    
    with tab2:
        # ポートフォリオタブ
        st.header("📊 Portfolio Overview")
        
        # ポートフォリオサマリー
        account = trader.get_account_summary()
        
        col_port1, col_port2, col_port3, col_port4 = st.columns(4)
        
        with col_port1:
            st.metric("Total Equity", f"${account['equity']:,.2f}")
        
        with col_port2:
            st.metric("Available Balance", f"${account['balance']:,.2f}")
        
        with col_port3:
            st.metric("Total Position Value", f"${account['total_position_value']:,.2f}")
        
        with col_port4:
            st.metric("Unrealized P&L", f"${account['unrealized_pnl']:,.2f}")
        
        # ポジション詳細
        positions = trader.get_positions()
        
        if positions:
            st.subheader("📍 Active Positions")
            
            position_data = []
            for symbol, pos in positions.items():
                pnl_emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                position_data.append({
                    'Symbol': symbol,
                    'Quantity': f"{pos['quantity']:.6f}",
                    'Entry Price': f"${pos['entry_price']:,.2f}",
                    'Current Price': f"${pos['current_price']:,.2f}",
                    'Market Value': f"${pos['market_value']:,.2f}",
                    'P&L': f"{pnl_emoji} ${pos['unrealized_pnl']:,.2f}",
                    'P&L %': f"{pos['unrealized_pnl_pct']:+.2f}%"
                })
            
            df_positions = pd.DataFrame(position_data)
            st.dataframe(df_positions, use_container_width=True, hide_index=True)
        else:
            st.info("No active positions")
        
        # パフォーマンスチャート
        st.subheader("📈 Performance Chart")
        
        # エクイティカーブ（シンプル版）
        equity_data = [account['equity']]  # 実際は履歴データを使用
        dates = [datetime.now()]
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=equity_data,
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#00D4AA', width=3)
        ))
        
        fig_equity.update_layout(
            title="Portfolio Value",
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with tab3:
        # 注文・履歴タブ
        st.header("📋 Orders & Trade History")
        
        # 最近の取引
        trades = trader.get_trade_history(20)
        
        if trades:
            st.subheader("📈 Recent Trades")
            
            trade_data = []
            for trade in reversed(trades[-10:]):  # 最新10件
                side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"
                trade_data.append({
                    'Time': trade['timestamp'][:19].replace('T', ' '),
                    'Symbol': trade['symbol'],
                    'Side': f"{side_emoji} {trade['side'].title()}",
                    'Quantity': f"{trade['quantity']:.6f}",
                    'Price': f"${trade['price']:,.2f}",
                    'Fee': f"${trade['fee']:.2f}",
                    'Slippage': f"{trade['slippage']:.3%}"
                })
            
            df_trades = pd.DataFrame(trade_data)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
            
            # 取引統計
            st.subheader("📊 Trading Statistics")
            
            buy_trades = len([t for t in trades if t['side'] == 'buy'])
            sell_trades = len([t for t in trades if t['side'] == 'sell'])
            total_fees = sum(t['fee'] for t in trades)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Trades", len(trades))
            
            with col_stat2:
                st.metric("Buy Orders", buy_trades)
            
            with col_stat3:
                st.metric("Sell Orders", sell_trades)
            
            with col_stat4:
                st.metric("Total Fees", f"${total_fees:.2f}")
        else:
            st.info("No trade history yet. Start trading to see your history here!")
    
    with tab4:
        # AI Insights タブ
        st.header("🤖 AI Market Insights")
        
        # AI予測結果
        try:
            ai_summary = ai_trader.get_enhanced_summary()
            predictions = ai_summary.get('latest_predictions', [])
            
            if predictions:
                st.subheader("🎯 AI Predictions")
                
                pred_data = []
                for pred in predictions:
                    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                    confidence_level = "High" if pred['confidence'] >= 0.7 else "Medium" if pred['confidence'] >= 0.5 else "Low"
                    
                    pred_data.append({
                        'Asset': pred['symbol'],
                        'Signal': f"{signal_emoji.get(pred['signal'], '🟡')} {pred['signal']}",
                        'Confidence': f"{pred['confidence']:.1%}",
                        'Level': confidence_level,
                        'Probability': f"{pred['probability']:.1%}",
                        'Current Price': f"${pred.get('price', 0):,.2f}"
                    })
                
                df_predictions = pd.DataFrame(pred_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                
                # 高信頼度の推奨
                high_conf_preds = [p for p in predictions if p['confidence'] >= 0.7]
                if high_conf_preds:
                    st.subheader("⭐ High Confidence Recommendations")
                    
                    for pred in high_conf_preds[:3]:
                        signal_color = "🟢" if pred['signal'] == 'BUY' else "🔴" if pred['signal'] == 'SELL' else "🟡"
                        
                        st.markdown(f"""
                        <div class="trading-card">
                            <h4>{signal_color} {pred['symbol']} - {pred['signal']} Signal</h4>
                            <p>Confidence: <strong>{pred['confidence']:.1%}</strong></p>
                            <p>Current Price: <strong>${pred.get('price', 0):,.2f}</strong></p>
                            <p>Upward Probability: <strong>{pred['probability']:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Run AI Prediction from the sidebar to see market insights")
                
        except Exception as e:
            st.error(f"AI insights error: {e}")
    
    # フッター
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; margin-top: 2rem;">
        📝 <strong>Realistic Paper Trading</strong> - This is a simulation using virtual funds. 
        No real money is at risk. | Session Trades: <strong>{}</strong>
    </div>
    """.format(st.session_state.total_trades), unsafe_allow_html=True)

if __name__ == "__main__":
    main()