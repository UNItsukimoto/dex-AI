#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
シンプルなトレーディングデモダッシュボード
応答性とユーザビリティに重点を置いた実装
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_ai_trader import EnhancedAITrader

# ページ設定
st.set_page_config(
    page_title="📈 Simple Trading Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def init_trader():
    """トレーダー初期化"""
    if 'trader' not in st.session_state:
        st.session_state.trader = EnhancedAITrader(10000.0)
        st.session_state.trade_count = 0
    return st.session_state.trader

def main():
    st.title("📈 Simple Trading Demo")
    st.markdown("**ペーパートレードでAI取引を体験**")
    
    # トレーダー初期化
    trader = init_trader()
    
    # サイドバー
    with st.sidebar:
        st.header("🎛️ コントロールパネル")
        
        # アカウント情報
        account = trader.trading_engine.get_account_summary()
        st.subheader("💰 アカウント")
        st.metric("残高", f"${account['balance']:,.0f}")
        st.metric("エクイティ", f"${account['equity']:,.0f}")
        
        profit_loss = account['equity'] - account['balance']
        color = "normal" if profit_loss >= 0 else "inverse"
        st.metric("損益", f"${profit_loss:,.0f}", delta_color=color)
        
        st.divider()
        
        # クイックアクション
        st.subheader("🚀 クイックアクション")
        
        if st.button("🎯 AI予測実行", use_container_width=True, key="quick_predict"):
            with st.spinner("AI予測実行中..."):
                try:
                    trader.execute_enhanced_strategy()
                    st.success("✅ 予測完了！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ エラー: {e}")
        
        if st.button("🔄 データ更新", use_container_width=True, key="quick_refresh"):
            st.rerun()
    
    # メインエリア
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 AI予測結果")
        
        # 予測結果表示
        try:
            summary = trader.get_enhanced_summary()
            predictions = summary.get('latest_predictions', [])
            
            if predictions:
                pred_data = []
                for pred in predictions:
                    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                    pred_data.append({
                        '銘柄': pred['symbol'],
                        '現在価格': f"${pred.get('price', 0):,.0f}",
                        'シグナル': f"{signal_emoji.get(pred['signal'], '🟡')} {pred['signal']}",
                        '信頼度': f"{pred['confidence']:.0%}",
                        '上昇確率': f"{pred['probability']:.0%}"
                    })
                
                df = pd.DataFrame(pred_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # 高信頼度の推奨表示
                high_conf_preds = [p for p in predictions if p['confidence'] >= 0.7]
                if high_conf_preds:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.write("**🎯 高信頼度推奨:**")
                    for pred in high_conf_preds[:2]:  # 上位2件
                        signal_color = "🟢" if pred['signal'] == 'BUY' else "🔴" if pred['signal'] == 'SELL' else "🟡"
                        st.write(f"{signal_color} **{pred['symbol']}** - {pred['signal']} (信頼度: {pred['confidence']:.0%})")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📝 AI予測を実行してください（サイドバーの「AI予測実行」ボタン）")
        
        except Exception as e:
            st.error(f"❌ 予測データ取得エラー: {e}")
    
    with col2:
        st.subheader("⚡ 簡単取引")
        
        # 銘柄選択
        symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
        selected_symbol = st.selectbox("銘柄", symbols)
        
        # 取引タイプ
        trade_type = st.radio("取引タイプ", ["買い", "売り"], horizontal=True)
        
        # 金額
        amount = st.number_input("金額 (USD)", min_value=100, max_value=5000, value=1000, step=100)
        
        # 取引実行ボタン
        if trade_type == "買い":
            if st.button(f"🚀 {selected_symbol} 買い注文", use_container_width=True, type="primary", key="buy_order"):
                try:
                    from core.paper_trading_engine import OrderSide, OrderType
                    
                    # 簡単な価格設定（実際の実装ではAPI価格を使用）
                    prices = {'BTC': 45000, 'ETH': 3200, 'SOL': 150, 'AVAX': 35}
                    price = prices.get(selected_symbol, 1000)
                    quantity = amount / price
                    
                    result = trader.trading_engine.place_order(
                        symbol=selected_symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=price
                    )
                    
                    if result:
                        st.success(f"✅ {selected_symbol} 買い注文成功！")
                        st.balloons()
                        st.session_state.trade_count += 1
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 注文失敗")
                        
                except Exception as e:
                    st.error(f"❌ 取引エラー: {e}")
        else:
            if st.button(f"📉 {selected_symbol} 売り注文", use_container_width=True, key="sell_order"):
                try:
                    from core.paper_trading_engine import OrderSide, OrderType
                    
                    prices = {'BTC': 45000, 'ETH': 3200, 'SOL': 150, 'AVAX': 35}
                    price = prices.get(selected_symbol, 1000)
                    quantity = amount / price
                    
                    result = trader.trading_engine.place_order(
                        symbol=selected_symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=price
                    )
                    
                    if result:
                        st.success(f"✅ {selected_symbol} 売り注文成功！")
                        st.session_state.trade_count += 1
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 注文失敗")
                        
                except Exception as e:
                    st.error(f"❌ 取引エラー: {e}")
        
        # 現在のポジション
        st.divider()
        st.subheader("📍 現在のポジション")
        
        positions = trader.trading_engine.get_positions()
        if positions:
            for symbol, pos in positions.items():
                st.write(f"**{symbol}**: {pos['quantity']:.4f}")
        else:
            st.info("ポジションなし")
    
    # 下部エリア
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📋 最近の取引")
        
        trades = trader.trading_engine.get_trade_history(limit=5)
        if trades:
            trade_data = []
            for trade in trades[-5:]:  # 最新5件
                trade_data.append({
                    '時刻': trade.get('timestamp', '')[:19],
                    '銘柄': trade.get('symbol', ''),
                    'タイプ': '買い' if trade.get('side') == 'buy' else '売り',
                    '数量': f"{trade.get('quantity', 0):.4f}",
                    '価格': f"${trade.get('price', 0):,.0f}"
                })
            
            df_trades = pd.DataFrame(trade_data)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.info("取引履歴なし")
    
    with col4:
        st.subheader("📊 取引統計")
        
        if trades:
            buy_trades = len([t for t in trades if t.get('side') == 'buy'])
            sell_trades = len([t for t in trades if t.get('side') == 'sell'])
            
            st.metric("総取引数", len(trades))
            st.metric("買い取引", buy_trades)
            st.metric("売り取引", sell_trades)
            st.metric("本セッション", st.session_state.trade_count)
        else:
            st.info("統計なし")
    
    # フッター
    st.divider()
    st.markdown("**📝 Note**: これはペーパートレード（仮想取引）です。実際の資金は使用されません。")

if __name__ == "__main__":
    main()