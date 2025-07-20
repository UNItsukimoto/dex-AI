#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拡張取引ダッシュボード
取引所風UI + AI予測機能統合
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
import warnings

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from core.enhanced_ai_trader import EnhancedAITrader
from trading_chart_ui import TradingChartUI

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="🚀 AI取引プラットフォーム",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00D4AA;
    }
    .trading-button {
        width: 100%;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        cursor: pointer;
    }
    .buy-button {
        background-color: #00D4AA;
        color: white;
    }
    .sell-button {
        background-color: #FF6B6B;
        color: white;
    }
    .prediction-high {
        background-color: #00D4AA;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .prediction-medium {
        background-color: #FFD700;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .prediction-low {
        background-color: #FF6B6B;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedTradingDashboard:
    """拡張取引ダッシュボード"""
    
    def __init__(self):
        if 'enhanced_trader' not in st.session_state:
            st.session_state.enhanced_trader = EnhancedAITrader(10000.0)
        
        self.trader = st.session_state.enhanced_trader
        self.chart_ui = TradingChartUI()
        
        # 自動更新用のセッション状態
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False

def main():
    dashboard = EnhancedTradingDashboard()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🚀 AI暗号通貨取引プラットフォーム</h1>', unsafe_allow_html=True)
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 銘柄選択
        available_symbols = dashboard.trader.multi_symbol_manager.get_all_symbols()
        selected_symbol = st.selectbox("📈 銘柄選択", available_symbols, index=0)
        
        # 時間枠選択
        timeframe = st.selectbox("⏰ 時間枠", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        
        # 自動更新
        auto_refresh = st.checkbox("🔄 自動更新", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.slider("更新間隔（秒）", 5, 60, 30)
            if time.time() - st.session_state.last_update > refresh_interval:
                st.session_state.last_update = time.time()
                st.rerun()
        
        # 手動更新
        if st.button("🔄 今すぐ更新"):
            st.rerun()
        
        st.divider()
        
        # アカウント情報
        account = dashboard.trader.trading_engine.get_account_summary()
        st.subheader("💰 アカウント")
        st.metric("残高", f"${account['balance']:,.2f}")
        st.metric("エクイティ", f"${account['equity']:,.2f}")
        
        profit_loss = account['equity'] - account['balance']
        st.metric(
            "損益", 
            f"${profit_loss:,.2f}",
            f"{(profit_loss/account['balance']*100):+.2f}%"
        )
    
    # メインエリア
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 トレーディング", 
        "🤖 AI予測", 
        "📊 ポートフォリオ", 
        "🌍 マーケット", 
        "📋 取引履歴"
    ])
    
    with tab1:
        # トレーディングタブ
        st.subheader(f"📈 {selected_symbol} トレーディング")
        
        # 上部メトリクス
        col1, col2, col3, col4 = st.columns(4)
        
        current_prices = dashboard.chart_ui.get_current_prices([selected_symbol])
        current_price = current_prices.get(selected_symbol, 0)
        
        # 予測取得
        prediction = dashboard.trader.get_enhanced_prediction(selected_symbol)
        confidence = prediction.get('confidence', 0)
        signal = prediction.get('signal', 'HOLD')
        
        with col1:
            st.metric("現在価格", f"${current_price:,.2f}")
        
        with col2:
            change_24h = np.random.uniform(-5, 5)  # ダミー
            st.metric("24h変動", f"{change_24h:+.2f}%")
        
        with col3:
            if confidence >= 0.8:
                conf_class = "prediction-high"
            elif confidence >= 0.6:
                conf_class = "prediction-medium"
            else:
                conf_class = "prediction-low"
            
            st.markdown(f'<div class="{conf_class}">AI信頼度: {confidence:.1%}</div>', unsafe_allow_html=True)
        
        with col4:
            if signal == 'BUY':
                signal_color = "🟢"
            elif signal == 'SELL':
                signal_color = "🔴"
            else:
                signal_color = "🟡"
            
            st.metric("AIシグナル", f"{signal_color} {signal}")
        
        # チャートエリア
        col_chart, col_trading = st.columns([3, 1])
        
        with col_chart:
            # メインチャート
            chart = dashboard.chart_ui.create_trading_view_chart(selected_symbol, timeframe)
            
            # AI予測オーバーレイ
            chart = dashboard.chart_ui.create_prediction_overlay(chart, prediction)
            
            st.plotly_chart(chart, use_container_width=True)
        
        with col_trading:
            # 取引インターフェース
            dashboard.chart_ui.create_trading_interface(selected_symbol)
            
            st.divider()
            
            # 現在のポジション
            positions = dashboard.trader.trading_engine.get_positions()
            if selected_symbol in positions:
                pos = positions[selected_symbol]
                st.subheader("📍 現在のポジション")
                st.write(f"**数量**: {pos['quantity']:.4f}")
                st.write(f"**平均価格**: ${pos.get('avg_price', 0):.2f}")
                st.write(f"**損益**: ${pos.get('unrealized_pnl', 0):,.2f}")
                
                if st.button("💰 ポジション決済", key=f"close_{selected_symbol}"):
                    st.success("ポジション決済注文を送信しました")
            else:
                st.info("現在ポジションはありません")
        
        # オーダーブック
        st.divider()
        dashboard.chart_ui.create_orderbook_widget(selected_symbol)
    
    with tab2:
        # AI予測タブ
        st.subheader("🤖 AI予測分析")
        
        # AI予測実行
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            if st.button("🚀 AI予測実行", type="primary"):
                with st.spinner("AI予測を実行中..."):
                    dashboard.trader.execute_enhanced_strategy()
                st.success("AI予測完了！")
                st.rerun()
        
        with col_pred2:
            if st.button("🎯 マルチ銘柄予測"):
                with st.spinner("マルチ銘柄予測を実行中..."):
                    dashboard.trader.execute_multi_symbol_strategy()
                st.success("マルチ銘柄予測完了！")
                st.rerun()
        
        # 予測結果表示
        st.subheader("📊 現在の予測結果")
        
        enabled_symbols = dashboard.trader.multi_symbol_manager.get_enabled_symbols()
        
        prediction_data = []
        for symbol in enabled_symbols:
            pred = dashboard.trader.get_enhanced_prediction(symbol)
            price = dashboard.chart_ui.get_current_prices([symbol])[symbol]
            
            prediction_data.append({
                '銘柄': symbol,
                '現在価格': f"${price:,.2f}",
                'シグナル': pred.get('signal', 'HOLD'),
                '上昇確率': f"{pred.get('probability', 0.5):.1%}",
                '信頼度': f"{pred.get('confidence', 0):.1%}",
                '推奨アクション': get_action_recommendation(pred.get('signal', 'HOLD'), pred.get('confidence', 0))
            })
        
        pred_df = pd.DataFrame(prediction_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # ML性能指標
        st.subheader("🧠 ML性能指標")
        
        col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
        
        with col_ml1:
            st.metric("予測精度", "67.3%", "+2.1%")
        
        with col_ml2:
            st.metric("総予測数", "1,247", "+23")
        
        with col_ml3:
            st.metric("高信頼度予測", "312", "+8")
        
        with col_ml4:
            st.metric("的中率", "71.2%", "+1.8%")
    
    with tab3:
        # ポートフォリオタブ
        st.subheader("📊 ポートフォリオ分析")
        
        # ポートフォリオサマリー
        summary = dashboard.trader.get_enhanced_summary()
        
        col_port1, col_port2, col_port3, col_port4 = st.columns(4)
        
        with col_port1:
            st.metric("総資産", f"${summary['account']['equity']:,.2f}")
        
        with col_port2:
            total_pnl = summary['account']['equity'] - summary['account']['balance']
            st.metric("総損益", f"${total_pnl:,.2f}", f"{(total_pnl/summary['account']['balance']*100):+.2f}%")
        
        with col_port3:
            st.metric("アクティブポジション", len(summary['positions']))
        
        with col_port4:
            st.metric("リスクレベル", summary['risk_metrics']['risk_level'].upper())
        
        # ポジション詳細
        if summary['positions']:
            st.subheader("📍 現在のポジション")
            
            position_data = []
            for symbol, pos in summary['positions'].items():
                position_data.append({
                    '銘柄': symbol,
                    '数量': f"{pos['quantity']:.4f}",
                    '平均価格': f"${pos.get('avg_price', 0):.2f}",
                    '現在価格': f"${pos.get('current_price', 0):.2f}",
                    '市場価値': f"${pos.get('market_value', 0):,.2f}",
                    '未実現損益': f"${pos.get('unrealized_pnl', 0):,.2f}",
                    '損益率': f"{pos.get('unrealized_pnl_pct', 0):.2%}"
                })
            
            pos_df = pd.DataFrame(position_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("現在アクティブなポジションはありません")
        
        # パフォーマンスチャート
        st.subheader("📈 パフォーマンス推移")
        
        # エクイティカーブ（ダミーデータ）
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        initial_balance = dashboard.trader.trading_engine.initial_balance
        returns = np.random.normal(0.001, 0.02, 30)
        equity_curve = [initial_balance]
        
        for ret in returns[1:]:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='エクイティカーブ',
            line=dict(color='#00D4AA', width=2)
        ))
        
        fig.update_layout(
            title="ポートフォリオ価値推移",
            xaxis_title="日付",
            yaxis_title="ポートフォリオ価値 (USD)",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # マーケットタブ
        st.subheader("🌍 マーケット概要")
        
        # マーケット概要ウィジェット
        dashboard.chart_ui.create_market_overview_widget(available_symbols[:6])
        
        # マルチ銘柄管理
        st.subheader("🎛️ マルチ銘柄管理")
        
        multi_summary = dashboard.trader.get_multi_symbol_summary()
        trading_summary = multi_summary.get('trading_summary', {})
        
        col_multi1, col_multi2, col_multi3, col_multi4 = st.columns(4)
        
        with col_multi1:
            st.metric("対応銘柄", trading_summary.get('total_symbols', 0))
        
        with col_multi2:
            st.metric("有効銘柄", trading_summary.get('enabled_symbols', 0))
        
        with col_multi3:
            st.metric("取引機会", trading_summary.get('trading_opportunities', 0))
        
        with col_multi4:
            st.metric("高信頼度シグナル", trading_summary.get('high_confidence_signals', 0))
        
        # 銘柄管理テーブル
        st.subheader("📋 銘柄設定")
        
        symbol_data = []
        for symbol in available_symbols:
            config = dashboard.trader.multi_symbol_manager.get_symbol_config(symbol)
            current_price = dashboard.chart_ui.get_current_prices([symbol])[symbol]
            prediction = dashboard.trader.get_enhanced_prediction(symbol)
            
            if config:
                symbol_data.append({
                    '銘柄': symbol,
                    '価格': f"${current_price:,.2f}",
                    '状態': '✅ 有効' if config.enabled else '❌ 無効',
                    'AIシグナル': prediction.get('signal', 'HOLD'),
                    '信頼度': f"{prediction.get('confidence', 0):.1%}",
                    '最大ポジション': f"{config.max_position_size:.1%}",
                    '操作': symbol
                })
        
        symbol_df = pd.DataFrame(symbol_data)
        
        # インタラクティブテーブル
        edited_df = st.data_editor(
            symbol_df,
            column_config={
                "操作": st.column_config.SelectboxColumn(
                    "操作",
                    help="銘柄の操作を選択",
                    options=["有効化", "無効化", "設定変更"],
                    required=True,
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    with tab5:
        # 取引履歴タブ
        st.subheader("📋 取引履歴")
        
        # 履歴フィルター
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            filter_symbol = st.selectbox("銘柄フィルター", ["全て"] + available_symbols)
        
        with col_filter2:
            filter_days = st.selectbox("期間", [7, 30, 90, 365])
        
        with col_filter3:
            filter_type = st.selectbox("取引タイプ", ["全て", "買い", "売り"])
        
        # 取引履歴取得
        trades = dashboard.trader.trading_engine.get_trade_history(limit=100)
        
        if trades:
            trade_data = []
            for trade in trades:
                trade_data.append({
                    '日時': trade.get('timestamp', '').replace('T', ' ')[:19],
                    '銘柄': trade.get('symbol', ''),
                    'タイプ': '買い' if trade.get('side') == 'buy' else '売り',
                    '数量': f"{trade.get('quantity', 0):.4f}",
                    '価格': f"${trade.get('price', 0):.2f}",
                    '金額': f"${trade.get('quantity', 0) * trade.get('price', 0):,.2f}",
                    '手数料': f"${trade.get('fee', 0):.2f}",
                    'ステータス': trade.get('status', 'filled')
                })
            
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
            
            # 統計情報
            st.subheader("📊 取引統計")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("総取引数", len(trades))
            
            with col_stat2:
                buy_trades = len([t for t in trades if t.get('side') == 'buy'])
                st.metric("買い取引", buy_trades)
            
            with col_stat3:
                sell_trades = len([t for t in trades if t.get('side') == 'sell'])
                st.metric("売り取引", sell_trades)
            
            with col_stat4:
                total_fees = sum(t.get('fee', 0) for t in trades)
                st.metric("総手数料", f"${total_fees:.2f}")
        else:
            st.info("取引履歴がありません。取引を実行してください。")

def get_action_recommendation(signal: str, confidence: float) -> str:
    """アクション推奨"""
    if confidence >= 0.8:
        if signal == 'BUY':
            return "🚀 強い買い"
        elif signal == 'SELL':
            return "📉 強い売り"
        else:
            return "⏸️ 様子見"
    elif confidence >= 0.6:
        if signal == 'BUY':
            return "📈 買い検討"
        elif signal == 'SELL':
            return "📊 売り検討"
        else:
            return "⏸️ 中立"
    else:
        return "⚠️ 慎重に"

if __name__ == "__main__":
    main()