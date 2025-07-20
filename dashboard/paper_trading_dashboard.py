#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ペーパートレーディングダッシュボード
AI予測ベースの仮想取引システム可視化
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

from core.ai_paper_trader import AIPaperTrader

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="ペーパートレーディングダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PaperTradingDashboard:
    """ペーパートレーディングダッシュボード"""
    
    def __init__(self):
        if 'ai_trader' not in st.session_state:
            st.session_state.ai_trader = AIPaperTrader(10000.0)
            # より積極的な設定
            st.session_state.ai_trader.confidence_threshold = 0.3  # 閾値を下げる
            st.session_state.ai_trader.buy_threshold = 0.6
            st.session_state.ai_trader.sell_threshold = 0.4
        
        self.ai_trader = st.session_state.ai_trader
    
    def create_account_chart(self, account_data):
        """アカウント情報チャート"""
        fig = go.Figure()
        
        # 円グラフ
        labels = ['使用可能資金', '使用中証拠金']
        values = [account_data['margin_free'], account_data['margin_used']]
        colors = ['lightgreen', 'lightcoral']
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.4
        ))
        
        fig.update_layout(
            title=f"資金状況 (総資産: ${account_data['equity']:,.2f})",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_predictions_chart(self, predictions):
        """予測確率チャート"""
        if not predictions:
            fig = go.Figure()
            fig.add_annotation(text="予測データなし", showarrow=False)
            return fig
        
        symbols = [p['symbol'] for p in predictions]
        probabilities = [p['probability'] * 100 for p in predictions]
        confidences = [p['confidence'] * 100 for p in predictions]
        
        # 色分け（シグナル別）
        colors = []
        for p in predictions:
            if p['signal'] == 'BUY':
                colors.append('green')
            elif p['signal'] == 'SELL':
                colors.append('red')
            else:
                colors.append('gray')
        
        fig = go.Figure()
        
        # 予測確率バー
        fig.add_trace(go.Bar(
            x=symbols,
            y=probabilities,
            name='上昇確率',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='auto'
        ))
        
        # 信頼度ライン
        fig.add_trace(go.Scatter(
            x=symbols,
            y=confidences,
            mode='lines+markers',
            name='信頼度',
            line=dict(color='orange', width=3),
            yaxis='y2'
        ))
        
        # 閾値ライン
        fig.add_hline(y=60, line_dash="dash", line_color="green", 
                     annotation_text="買い閾値")
        fig.add_hline(y=40, line_dash="dash", line_color="red", 
                     annotation_text="売り閾値")
        
        fig.update_layout(
            title="AI予測確率とシグナル",
            xaxis_title="銘柄",
            yaxis_title="確率 (%)",
            yaxis2=dict(title="信頼度 (%)", overlaying='y', side='right'),
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_pnl_chart(self, positions, trades):
        """PnLチャート"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ポジション別未実現PnL', '取引履歴'),
            vertical_spacing=0.1
        )
        
        # ポジション別PnL
        if positions:
            symbols = list(positions.keys())
            pnls = [positions[s]['unrealized_pnl'] for s in symbols]
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=pnls,
                marker_color=colors,
                name='未実現PnL',
                text=[f"${pnl:.2f}" for pnl in pnls],
                textposition='auto'
            ), row=1, col=1)
        
        # 取引履歴
        if trades:
            trade_times = [datetime.fromisoformat(t['timestamp']) for t in trades]
            trade_pnls = [t['pnl'] for t in trades]
            cumulative_pnl = np.cumsum(trade_pnls)
            
            fig.add_trace(go.Scatter(
                x=trade_times,
                y=cumulative_pnl,
                mode='lines+markers',
                name='累積PnL',
                line=dict(color='blue', width=2)
            ), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_trading_signals_chart(self, signals):
        """取引シグナルチャート"""
        if not signals:
            fig = go.Figure()
            fig.add_annotation(text="取引シグナルなし", showarrow=False)
            return fig
        
        # 最新50シグナル
        recent_signals = signals[-50:] if len(signals) > 50 else signals
        
        times = [datetime.fromisoformat(s['timestamp']) for s in recent_signals]
        symbols = [s['symbol'] for s in recent_signals]
        actions = [s['action'] for s in recent_signals]
        prices = [s['price'] for s in recent_signals]
        
        fig = go.Figure()
        
        # 銘柄別にプロット
        unique_symbols = list(set(symbols))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, symbol in enumerate(unique_symbols):
            symbol_data = [(t, p, a) for t, s, p, a in zip(times, symbols, prices, actions) if s == symbol]
            if symbol_data:
                symbol_times, symbol_prices, symbol_actions = zip(*symbol_data)
                
                # BUYとSELLで分けてプロット
                buy_times = [t for t, a in zip(symbol_times, symbol_actions) if a == 'BUY']
                buy_prices = [p for p, a in zip(symbol_prices, symbol_actions) if a == 'BUY']
                sell_times = [t for t, a in zip(symbol_times, symbol_actions) if a == 'SELL']
                sell_prices = [p for p, a in zip(symbol_prices, symbol_actions) if a == 'SELL']
                
                if buy_times:
                    fig.add_trace(go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name=f'{symbol} BUY',
                        marker=dict(symbol='triangle-up', size=10, color=colors[i % len(colors)])
                    ))
                
                if sell_times:
                    fig.add_trace(go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name=f'{symbol} SELL',
                        marker=dict(symbol='triangle-down', size=10, color=colors[i % len(colors)])
                    ))
        
        fig.update_layout(
            title="取引シグナル履歴",
            xaxis_title="時刻",
            yaxis_title="価格 (USD)",
            template='plotly_dark',
            height=400
        )
        
        return fig

def main():
    """メインアプリケーション"""
    st.title("📊 ペーパートレーディングダッシュボード")
    st.markdown("### AI予測ベースの仮想取引システム")
    
    dashboard = PaperTradingDashboard()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 取引設定")
    
    # 自動取引設定
    auto_trading = st.sidebar.checkbox("自動取引", value=False)
    trading_interval = st.sidebar.slider("取引間隔 (分)", 1, 30, 5)
    
    # 手動実行ボタン
    if st.sidebar.button("手動取引実行", type="primary"):
        with st.spinner("AI取引戦略実行中..."):
            dashboard.ai_trader.execute_trading_strategy()
        st.success("取引戦略実行完了")
        st.rerun()
    
    # リセットボタン
    if st.sidebar.button("システムリセット", help="全ポジションと履歴をリセット"):
        if st.sidebar.checkbox("リセット確認"):
            st.session_state.ai_trader = AIPaperTrader(10000.0)
            st.session_state.ai_trader.confidence_threshold = 0.3
            st.session_state.ai_trader.buy_threshold = 0.6
            st.session_state.ai_trader.sell_threshold = 0.4
            st.success("システムをリセットしました")
            st.rerun()
    
    # 取引設定調整
    st.sidebar.subheader("AI設定")
    new_confidence = st.sidebar.slider("信頼度閾値", 0.1, 0.9, dashboard.ai_trader.confidence_threshold)
    new_buy_threshold = st.sidebar.slider("買い閾値", 0.5, 0.8, dashboard.ai_trader.buy_threshold)
    new_sell_threshold = st.sidebar.slider("売り閾値", 0.2, 0.5, dashboard.ai_trader.sell_threshold)
    
    # 設定更新
    if (new_confidence != dashboard.ai_trader.confidence_threshold or 
        new_buy_threshold != dashboard.ai_trader.buy_threshold or 
        new_sell_threshold != dashboard.ai_trader.sell_threshold):
        dashboard.ai_trader.confidence_threshold = new_confidence
        dashboard.ai_trader.buy_threshold = new_buy_threshold
        dashboard.ai_trader.sell_threshold = new_sell_threshold
        st.sidebar.success("設定更新")
    
    # メインコンテンツ
    # アカウント情報表示
    summary = dashboard.ai_trader.get_trading_summary()
    account = summary['account']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="総資産",
            value=f"${account['equity']:,.2f}",
            delta=f"${account['total_pnl']:+.2f}"
        )
    
    with col2:
        st.metric(
            label="リターン",
            value=f"{account['return_pct']:+.2f}%",
            delta=f"取引数: {account['total_trades']}"
        )
    
    with col3:
        st.metric(
            label="勝率",
            value=f"{account['win_rate']:.1f}%",
            delta=f"勝利: {account['winning_trades']}"
        )
    
    with col4:
        st.metric(
            label="アクティブポジション",
            value=len(summary['positions']),
            delta=f"シグナル: {summary['signal_count']}"
        )
    
    # メインチャートエリア
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # 予測とシグナル
        st.subheader("🎯 AI予測とシグナル")
        
        predictions = dashboard.ai_trader.get_latest_predictions()
        if predictions:
            pred_chart = dashboard.create_predictions_chart(predictions)
            st.plotly_chart(pred_chart, use_container_width=True)
            
            # 予測詳細テーブル
            pred_data = []
            for pred in predictions:
                pred_data.append({
                    '銘柄': pred['symbol'],
                    '現在価格': f"${pred['price']:.2f}",
                    '上昇確率': f"{pred['probability']:.1%}",
                    'シグナル': pred['signal'],
                    '信頼度': f"{pred['confidence']:.1%}",
                    'RSI': f"{pred['features'].get('rsi', 0):.1f}" if pred['features'] else "N/A"
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("予測データを取得するには「手動取引実行」ボタンを押してください")
    
    with col_right:
        # アカウント状況
        st.subheader("💰 資金状況")
        
        if account['equity'] > 0:
            account_chart = dashboard.create_account_chart(account)
            st.plotly_chart(account_chart, use_container_width=True)
        
        # 現在ポジション
        st.subheader("📈 現在ポジション")
        if summary['positions']:
            for symbol, pos in summary['positions'].items():
                pnl_color = "green" if pos['unrealized_pnl'] >= 0 else "red"
                st.markdown(f"""
                **{symbol}**  
                {pos['side'].upper()} {pos['quantity']}  
                エントリー: ${pos['entry_price']:.2f}  
                現在: ${pos['current_price']:.2f}  
                :{pnl_color}[PnL: ${pos['unrealized_pnl']:+.2f}]
                """)
        else:
            st.info("現在ポジションなし")
    
    # 下部セクション
    st.markdown("---")
    
    col_pnl, col_signals = st.columns(2)
    
    with col_pnl:
        st.subheader("📊 PnL分析")
        
        pnl_chart = dashboard.create_pnl_chart(summary['positions'], summary['recent_trades'])
        st.plotly_chart(pnl_chart, use_container_width=True)
    
    with col_signals:
        st.subheader("📡 取引シグナル履歴")
        
        signals_chart = dashboard.create_trading_signals_chart(dashboard.ai_trader.trade_signals)
        st.plotly_chart(signals_chart, use_container_width=True)
    
    # 取引履歴
    if summary['recent_trades']:
        st.subheader("📋 最近の取引")
        
        trade_data = []
        for trade in summary['recent_trades'][-10:]:  # 最新10件
            trade_data.append({
                '時刻': trade['timestamp'][:19],
                '銘柄': trade['symbol'],
                'サイド': trade['side'].upper(),
                '数量': trade['quantity'],
                '価格': f"${trade['price']:.2f}",
                '手数料': f"${trade['fee']:.2f}",
                'PnL': f"${trade['pnl']:+.2f}" if trade['pnl'] != 0 else "-"
            })
        
        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True)
    
    # フッター情報
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.caption("🤖 AI予測エンジン: 改良版50%超システム")
        st.caption("💼 初期資金: $10,000")
    
    with col_info2:
        st.caption("⚡ 取引手数料: 0.1%")
        st.caption("📊 最大ポジション: 20%")
    
    with col_info3:
        st.caption("🎯 対象銘柄: BTC, ETH, SOL, AVAX")
        st.caption("🔄 リアルタイム価格: Hyperliquid")
    
    # 自動取引
    if auto_trading:
        time.sleep(trading_interval * 60)
        dashboard.ai_trader.execute_trading_strategy()
        st.rerun()

if __name__ == "__main__":
    main()