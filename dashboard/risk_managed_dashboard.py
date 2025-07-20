#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リスク管理統合ダッシュボード
高度なリスク制御機能付きAI取引システム
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

from core.enhanced_ai_trader import EnhancedAITrader
from trading_chart_ui import TradingChartUI

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="リスク管理統合ダッシュボード",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RiskManagedDashboard:
    """リスク管理統合ダッシュボード"""
    
    def __init__(self):
        if 'enhanced_trader' not in st.session_state:
            st.session_state.enhanced_trader = EnhancedAITrader(10000.0)
        
        self.trader = st.session_state.enhanced_trader
        self.chart_ui = TradingChartUI()
    
    def create_risk_gauge(self, risk_level: str, risk_score: float) -> go.Figure:
        """リスクレベルゲージ"""
        risk_colors = {
            'low': 'green',
            'medium': 'yellow', 
            'high': 'orange',
            'extreme': 'red'
        }
        
        risk_values = {
            'low': 25,
            'medium': 50,
            'high': 75,
            'extreme': 100
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_values.get(risk_level, 25),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "リスクレベル"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_colors.get(risk_level, 'gray')},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, template='plotly_dark')
        return fig
    
    def create_exposure_chart(self, positions: dict, account_equity: float) -> go.Figure:
        """エクスポージャーチャート"""
        if not positions:
            fig = go.Figure()
            fig.add_annotation(text="ポジションなし", showarrow=False)
            fig.update_layout(template='plotly_dark', height=300)
            return fig
        
        symbols = list(positions.keys())
        values = [abs(pos['quantity'] * pos['current_price']) for pos in positions.values()]
        percentages = [v / account_equity * 100 for v in values]
        
        # 色分け（リスクレベル別）
        colors = []
        for pct in percentages:
            if pct > 20:
                colors.append('red')
            elif pct > 15:
                colors.append('orange')
            elif pct > 10:
                colors.append('yellow')
            else:
                colors.append('green')
        
        fig = go.Figure(data=[
            go.Bar(x=symbols, y=percentages, marker_color=colors,
                   text=[f"{p:.1f}%" for p in percentages],
                   textposition='auto')
        ])
        
        fig.add_hline(y=15, line_dash="dash", line_color="orange", 
                     annotation_text="推奨上限 15%")
        fig.add_hline(y=10, line_dash="dash", line_color="yellow", 
                     annotation_text="通常上限 10%")
        
        fig.update_layout(
            title="ポジション別エクスポージャー",
            xaxis_title="銘柄",
            yaxis_title="エクスポージャー (%)",
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def create_risk_metrics_chart(self, risk_metrics: dict) -> go.Figure:
        """リスク指標チャート"""
        metrics = ['総エクスポージャー', 'ドローダウン', 'ポートフォリオVol', '集中リスク']
        values = [
            risk_metrics.get('total_exposure', 0) * 100,
            risk_metrics.get('max_drawdown', 0) * 100,
            risk_metrics.get('portfolio_volatility', 0) * 100,
            risk_metrics.get('concentration_risk', 0) * 100
        ]
        limits = [60, 20, 50, 15]  # 制限値
        
        fig = go.Figure()
        
        # 現在値
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            name='現在値',
            marker_color=['red' if v > l else 'orange' if v > l*0.8 else 'green' 
                         for v, l in zip(values, limits)]
        ))
        
        # 制限値ライン
        for i, (metric, limit) in enumerate(zip(metrics, limits)):
            fig.add_shape(
                type="line",
                x0=i-0.4, x1=i+0.4,
                y0=limit, y1=limit,
                line=dict(color="red", width=2, dash="dash")
            )
        
        fig.update_layout(
            title="リスク指標 vs 制限値",
            xaxis_title="指標",
            yaxis_title="値 (%)",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_prediction_analysis_chart(self, predictions: list) -> go.Figure:
        """予測分析チャート"""
        if not predictions:
            fig = go.Figure()
            fig.add_annotation(text="予測データなし", showarrow=False)
            fig.update_layout(template='plotly_dark', height=400)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('上昇確率', '信頼度', '予測成分', 'シグナル分布'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        symbols = [p['symbol'] for p in predictions]
        probabilities = [p['probability'] * 100 for p in predictions]
        confidences = [p['confidence'] * 100 for p in predictions]
        
        # 上昇確率
        fig.add_trace(go.Bar(x=symbols, y=probabilities, name='上昇確率',
                            marker_color=['green' if p > 60 else 'red' if p < 40 else 'gray' 
                                        for p in probabilities]),
                     row=1, col=1)
        
        # 信頼度
        fig.add_trace(go.Bar(x=symbols, y=confidences, name='信頼度',
                            marker_color='blue'),
                     row=1, col=2)
        
        # 予測成分（第1銘柄）
        if predictions and 'prediction_components' in predictions[0]:
            components = predictions[0]['prediction_components']
            comp_names = list(components.keys())
            comp_values = [v * 100 for v in components.values()]
            
            fig.add_trace(go.Bar(x=comp_names, y=comp_values, name='予測成分',
                                marker_color=['green' if v > 0 else 'red' for v in comp_values]),
                         row=2, col=1)
        
        # シグナル分布
        signals = [p['signal'] for p in predictions]
        signal_counts = {s: signals.count(s) for s in set(signals)}
        
        fig.add_trace(go.Pie(labels=list(signal_counts.keys()), 
                            values=list(signal_counts.values()),
                            name="シグナル"),
                     row=2, col=2)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_performance_chart(self, performance_stats: dict, account_history: list) -> go.Figure:
        """パフォーマンスチャート"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('資産推移', '勝敗統計', '取引統計', 'ドローダウン'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 資産推移（仮想データ）
        if account_history:
            times = [datetime.now() - timedelta(hours=i) for i in range(len(account_history), 0, -1)]
            equities = account_history
        else:
            times = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
            equities = [10000 + np.random.normal(0, 50) for _ in times]
        
        fig.add_trace(go.Scatter(x=times, y=equities, name='資産推移',
                                line=dict(color='blue', width=2)),
                     row=1, col=1)
        
        # 勝敗統計
        wins = performance_stats.get('successful_trades', 0)
        losses = performance_stats.get('failed_trades', 0)
        
        fig.add_trace(go.Bar(x=['勝利', '敗北'], y=[wins, losses],
                            marker_color=['green', 'red'], name='勝敗'),
                     row=1, col=2)
        
        # 取引統計
        total_signals = performance_stats.get('total_signals', 0)
        max_wins = performance_stats.get('max_consecutive_wins', 0)
        max_losses = performance_stats.get('max_consecutive_losses', 0)
        
        fig.add_trace(go.Bar(x=['総シグナル', '最大連勝', '最大連敗'], 
                            y=[total_signals, max_wins, max_losses],
                            marker_color=['blue', 'green', 'red'], name='統計'),
                     row=2, col=1)
        
        # ドローダウン（仮想データ）
        drawdowns = [max(0, 10000 - eq) / 10000 * 100 for eq in equities]
        fig.add_trace(go.Scatter(x=times, y=drawdowns, name='ドローダウン',
                                fill='tozeroy', line=dict(color='red')),
                     row=2, col=2)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig

def main():
    """メインアプリケーション"""
    st.title("🛡️ リスク管理統合ダッシュボード")
    st.markdown("### 高度なリスク制御機能付きAI取引システム")
    
    dashboard = RiskManagedDashboard()
    
    # サイドバー設定
    st.sidebar.header("🎛️ リスク設定")
    
    # リスク制限設定
    st.sidebar.subheader("制限値")
    max_position = st.sidebar.slider("最大ポジション (%)", 5, 25, 15)
    max_exposure = st.sidebar.slider("最大エクスポージャー (%)", 30, 80, 60)
    max_drawdown = st.sidebar.slider("最大ドローダウン (%)", 10, 30, 20)
    
    # AI設定
    st.sidebar.subheader("AI設定")
    confidence_threshold = st.sidebar.slider("信頼度閾値", 0.2, 0.8, 0.4)
    buy_threshold = st.sidebar.slider("買い閾値", 0.55, 0.75, 0.65)
    sell_threshold = st.sidebar.slider("売り閾値", 0.25, 0.45, 0.35)
    
    # 設定更新
    dashboard.trader.confidence_threshold = confidence_threshold
    dashboard.trader.buy_threshold = buy_threshold
    dashboard.trader.sell_threshold = sell_threshold
    
    # 手動実行
    col_exec1, col_exec2 = st.sidebar.columns(2)
    
    with col_exec1:
        if st.button("取引実行", type="primary"):
            with st.spinner("強化AI取引実行中..."):
                dashboard.trader.execute_enhanced_strategy()
            st.success("実行完了")
            st.rerun()
    
    with col_exec2:
        if st.button("緊急停止", type="secondary"):
            st.warning("全ポジション決済機能は実装予定")
    
    # メインコンテンツ
    # 上部ステータス
    summary = dashboard.trader.get_advanced_summary()  # 高度サマリーに変更
    account = summary['account']
    risk_metrics = summary['risk_metrics']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
            delta=f"勝率: {account['win_rate']:.1f}%"
        )
    
    with col3:
        risk_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "extreme": "🔴"}
        st.metric(
            label="リスクレベル",
            value=f"{risk_color.get(risk_metrics['risk_level'], '⚪')} {risk_metrics['risk_level'].upper()}",
            delta=f"エクスポージャー: {risk_metrics['total_exposure']:.1%}"
        )
    
    with col4:
        st.metric(
            label="シャープレシオ",
            value=f"{risk_metrics['sharpe_ratio']:.2f}",
            delta=f"VaR: {risk_metrics['var_95']:.1%}"
        )
    
    with col5:
        st.metric(
            label="アクティブポジション",
            value=len(summary['positions']),
            delta=f"シグナル: {summary['total_signals']}"
        )
    
    # リスク警告
    violations = summary['risk_violations']
    if violations:
        st.error(f"⚠️ {len(violations)}件のリスク違反が検出されました")
        for violation in violations[:3]:
            st.warning(f"**{violation['severity'].upper()}**: {violation['message']}")
    
    # メインダッシュボードエリア
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 トレーディング", "📊 リスク分析", "🎯 予測分析", "🤖 ML性能", "🚨 アラート", "📈 パフォーマンス", "🌍 マルチ銘柄", "⚙️ 詳細設定"])
    
    with tab1:
        # 取引所風トレーディングタブ
        st.header("📈 リアルタイム取引")
        
        # 銘柄選択
        col_select1, col_select2, col_select3 = st.columns([1, 1, 1])
        
        with col_select1:
            available_symbols = dashboard.trader.multi_symbol_manager.get_all_symbols()
            selected_symbol = st.selectbox("銘柄選択", available_symbols, key="main_symbol")
        
        with col_select2:
            timeframe = st.selectbox("時間枠", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3, key="main_timeframe")
        
        with col_select3:
            if st.button("🔄 チャート更新", key="refresh_chart"):
                st.rerun()
        
        # 現在価格と予測情報
        current_prices = dashboard.chart_ui.get_current_prices([selected_symbol])
        current_price = current_prices.get(selected_symbol, 0)
        prediction = dashboard.trader.get_enhanced_prediction(selected_symbol)
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.metric("💰 現在価格", f"${current_price:,.2f}")
        
        with col_info2:
            confidence = prediction.get('confidence', 0)
            st.metric("🎯 AI信頼度", f"{confidence:.1%}")
        
        with col_info3:
            signal = prediction.get('signal', 'HOLD')
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            st.metric("📊 AIシグナル", f"{signal_emoji.get(signal, '🟡')} {signal}")
        
        with col_info4:
            probability = prediction.get('probability', 0.5)
            st.metric("📈 上昇確率", f"{probability:.1%}")
        
        # メインチャートエリア
        col_chart, col_trading = st.columns([3, 1])
        
        with col_chart:
            st.subheader(f"📊 {selected_symbol} 価格チャート")
            
            # TradingView風チャート
            chart = dashboard.chart_ui.create_trading_view_chart(selected_symbol, timeframe)
            
            # AI予測オーバーレイ
            if prediction:
                chart = dashboard.chart_ui.create_prediction_overlay(chart, prediction)
            
            st.plotly_chart(chart, use_container_width=True)
        
        with col_trading:
            # 取引パネル
            st.subheader("⚡ 取引パネル")
            
            # 簡易取引インターフェース
            trade_type = st.radio("取引タイプ", ["買い", "売り"], horizontal=True)
            
            amount = st.number_input("金額 (USD)", min_value=10.0, value=100.0, step=10.0)
            
            if trade_type == "買い":
                if st.button("🚀 買い注文", type="primary", use_container_width=True):
                    st.success(f"✅ {selected_symbol} 買い注文 ${amount:.2f}")
                    st.balloons()
            else:
                if st.button("📉 売り注文", type="secondary", use_container_width=True):
                    st.success(f"✅ {selected_symbol} 売り注文 ${amount:.2f}")
            
            st.divider()
            
            # AI推奨
            st.subheader("🤖 AI推奨")
            
            if confidence >= 0.8:
                if signal == "BUY":
                    st.success("🚀 **強い買い推奨**")
                elif signal == "SELL":
                    st.error("📉 **強い売り推奨**")
                else:
                    st.info("⏸️ **様子見推奨**")
            elif confidence >= 0.6:
                if signal == "BUY":
                    st.info("📈 買い検討")
                elif signal == "SELL":
                    st.warning("📊 売り検討")
                else:
                    st.info("⏸️ 中立")
            else:
                st.warning("⚠️ 慎重に判断")
            
            # リスク情報
            st.subheader("⚠️ リスク情報")
            config = dashboard.trader.multi_symbol_manager.get_symbol_config(selected_symbol)
            if config:
                st.write(f"**最大ポジション**: {config.max_position_size:.1%}")
                st.write(f"**ストップロス**: {config.stop_loss_pct:.1%}")
                st.write(f"**利確目標**: {config.take_profit_pct:.1%}")
        
        # オーダーブック
        st.divider()
        dashboard.chart_ui.create_orderbook_widget(selected_symbol)
    
    with tab2:
        col_risk1, col_risk2 = st.columns([1, 2])
        
        with col_risk1:
            st.subheader("リスクレベル")
            risk_gauge = dashboard.create_risk_gauge(
                risk_metrics['risk_level'], 
                risk_metrics['total_exposure']
            )
            st.plotly_chart(risk_gauge, use_container_width=True)
        
        with col_risk2:
            st.subheader("リスク指標")
            risk_chart = dashboard.create_risk_metrics_chart(risk_metrics)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        # エクスポージャー分析
        st.subheader("ポジション分析")
        exposure_chart = dashboard.create_exposure_chart(summary['positions'], account['equity'])
        st.plotly_chart(exposure_chart, use_container_width=True)
        
        # 現在ポジション詳細
        if summary['positions']:
            st.subheader("現在ポジション詳細")
            pos_data = []
            for symbol, pos in summary['positions'].items():
                exposure_pct = abs(pos['quantity'] * pos['current_price']) / account['equity'] * 100
                pos_data.append({
                    '銘柄': symbol,
                    'サイド': pos['side'].upper(),
                    '数量': pos['quantity'],
                    'エントリー価格': f"${pos['entry_price']:.2f}",
                    '現在価格': f"${pos['current_price']:.2f}",
                    'エクスポージャー': f"{exposure_pct:.1f}%",
                    '未実現PnL': f"${pos['unrealized_pnl']:+.2f}"
                })
            
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True)
    
    with tab2:
        st.subheader("AI予測分析")
        
        predictions = summary['latest_predictions']
        if predictions:
            pred_chart = dashboard.create_prediction_analysis_chart(predictions)
            st.plotly_chart(pred_chart, use_container_width=True)
            
            # 予測詳細テーブル
            st.subheader("予測詳細")
            pred_data = []
            for pred in predictions:
                pred_data.append({
                    '銘柄': pred['symbol'],
                    '現在価格': f"${pred.get('price', 0):.2f}",
                    '上昇確率': f"{pred['probability']:.1%}",
                    'シグナル': pred['signal'],
                    '信頼度': f"{pred['confidence']:.1%}",
                    '更新時刻': pred.get('timestamp', '')[:19]
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("予測データを取得するには「取引実行」ボタンを押してください")
    
    with tab3:
        st.subheader("🤖 機械学習性能分析")
        
        # ML性能データ取得
        ml_performance = summary.get('ml_performance', {})
        prediction_stats = summary.get('prediction_stats', {})
        
        if ml_performance:
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                st.write("**モデル性能**")
                model_perf = ml_performance.get('model_performance', {})
                if model_perf:
                    for model_name, perf in model_perf.items():
                        accuracy = perf.get('accuracy', 0)
                        cv_score = perf.get('cv_mean', 0)
                        st.metric(
                            f"{model_name}",
                            f"{accuracy:.1%}",
                            f"CV: {cv_score:.1%}"
                        )
                else:
                    st.info("モデル学習データ不足（200サンプル必要）")
                
                # 強制再学習ボタン
                if st.button("🔄 モデル再学習", help="機械学習モデルを強制的に再学習"):
                    with st.spinner("モデル再学習中..."):
                        dashboard.trader.force_model_retrain()
                    st.success("再学習完了")
                    st.rerun()
            
            with col_ml2:
                st.write("**予測統計**")
                if prediction_stats:
                    st.metric("総予測回数", prediction_stats.get('total_predictions', 0))
                    st.metric("平均信頼度", f"{prediction_stats.get('average_confidence', 0):.1%}")
                    st.metric("高信頼度率", f"{prediction_stats.get('high_confidence_rate', 0):.1%}")
                    st.metric("学習サンプル数", ml_performance.get('training_samples', 0))
                
                st.write("**特徴量重要度**")
                feature_importance = ml_performance.get('feature_importance', {})
                if feature_importance:
                    # 最も性能の良いモデルの重要度表示
                    best_model = max(model_perf.keys(), key=lambda k: model_perf[k].get('accuracy', 0)) if model_perf else None
                    if best_model and best_model in feature_importance:
                        importance_data = feature_importance[best_model]
                        top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        for feature, importance in top_features:
                            st.write(f"{feature}: {importance:.3f}")
                else:
                    st.info("特徴量重要度データなし")
        else:
            st.info("機械学習性能データがありません。まず取引を実行してください。")
    
    with tab4:
        st.subheader("🚨 アラート・通知システム")
        
        # アラートシステム情報取得
        alert_summary = dashboard.trader.get_alert_system_summary()
        
        col_alert1, col_alert2 = st.columns(2)
        
        with col_alert1:
            st.write("**アラート統計（24時間）**")
            alert_stats = alert_summary.get('alert_stats', {})
            if alert_stats:
                st.metric("総アラート数", alert_stats.get('total_alerts', 0))
                
                # タイプ別統計
                by_type = alert_stats.get('by_type', {})
                if by_type:
                    st.write("**タイプ別:**")
                    for alert_type, count in by_type.items():
                        st.write(f"- {alert_type}: {count}件")
                
                # 優先度別統計
                by_priority = alert_stats.get('by_priority', {})
                if by_priority:
                    st.write("**優先度別:**")
                    for priority, count in by_priority.items():
                        emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        st.write(f"- {emoji} {priority.upper()}: {count}件")
            else:
                st.info("アラート統計データなし")
            
            # テストアラート送信
            if st.button("🧪 テストアラート送信", help="通知システムの動作確認"):
                with st.spinner("テストアラート送信中..."):
                    results = dashboard.trader.send_test_alert()
                
                st.write("**送信結果:**")
                for channel, success in results.items():
                    status = "✅ 成功" if success else "❌ 失敗"
                    st.write(f"- {channel}: {status}")
        
        with col_alert2:
            st.write("**通知チャンネル設定**")
            
            # 有効チャンネル表示
            enabled_channels = alert_summary.get('enabled_channels', [])
            total_channels = alert_summary.get('channels_count', 0)
            
            st.metric("有効チャンネル", f"{len(enabled_channels)}/{total_channels}")
            
            for channel in enabled_channels:
                icon = {"Desktop": "🖥️", "Audio": "🔊", "Email": "📧", "Slack": "💬"}.get(channel, "📢")
                st.write(f"{icon} {channel}")
            
            # アラート設定
            st.write("**アラート閾値**")
            
            col_threshold1, col_threshold2 = st.columns(2)
            with col_threshold1:
                confidence_threshold = st.slider("信頼度閾値", 0.5, 1.0, 0.8, 0.05, 
                                                help="この値以上でアラート送信")
                price_change_threshold = st.slider("価格変動閾値", 0.01, 0.2, 0.05, 0.01,
                                                  help="この変動率以上でアラート送信", format="%.2f")
            
            with col_threshold2:
                risk_score_threshold = st.slider("リスクスコア閾値", 0.5, 1.0, 0.8, 0.05,
                                                help="この値以上でリスクアラート送信")
        
        # 最近のアラート履歴
        st.subheader("📜 最近のアラート履歴")
        
        alert_history = alert_summary.get('alert_history', [])
        if alert_history:
            # 最新10件表示
            recent_alerts = alert_history[:10]
            
            alert_data = []
            for alert in recent_alerts:
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }.get(alert.priority.value, "⚪")
                
                type_emoji = {
                    "trading_signal": "📈",
                    "risk_warning": "⚠️",
                    "price_alert": "💰",
                    "system_error": "🚫",
                    "performance": "📊"
                }.get(alert.alert_type.value, "📢")
                
                alert_data.append({
                    "時刻": alert.timestamp.strftime("%H:%M:%S"),
                    "優先度": f"{priority_emoji} {alert.priority.value.upper()}",
                    "タイプ": f"{type_emoji} {alert.alert_type.value}",
                    "タイトル": alert.title,
                    "銘柄": alert.symbol or "-"
                })
            
            alert_df = pd.DataFrame(alert_data)
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
        else:
            st.info("アラート履歴なし。取引を実行するとアラートが生成されます。")
    
    with tab5:
        st.subheader("📊 パフォーマンス分析・レポート")
        
        # パフォーマンスサマリー取得
        performance_summary = dashboard.trader.get_performance_summary()
        
        if performance_summary and performance_summary.get('metrics'):
            metrics = performance_summary['metrics']
            
            # 主要指標表示
            col_main1, col_main2, col_main3, col_main4 = st.columns(4)
            
            with col_main1:
                st.metric(
                    "総損益",
                    f"${metrics['total_return']:,.2f}",
                    f"{metrics['total_return_pct']:+.1f}%"
                )
            
            with col_main2:
                st.metric(
                    "勝率",
                    f"{metrics['win_rate']:.1f}%",
                    f"取引数: {metrics['total_trades']}"
                )
            
            with col_main3:
                st.metric(
                    "シャープレシオ",
                    f"{metrics['sharpe_ratio']:.2f}",
                    "リスク調整後リターン"
                )
            
            with col_main4:
                st.metric(
                    "最大DD",
                    f"{metrics['max_drawdown']:.1f}%",
                    "最大ドローダウン",
                    delta_color="inverse"
                )
            
            # チャート表示
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # 既存の簡易チャート
                perf_chart = dashboard.create_performance_chart(
                    summary['performance_stats'], 
                    [account['equity']]
                )
                st.plotly_chart(perf_chart, use_container_width=True)
            
            with col_chart2:
                # 銘柄別パフォーマンス
                st.write("**銘柄別パフォーマンス**")
                symbol_perfs = performance_summary.get('symbol_performance', [])
                if symbol_perfs:
                    symbol_data = []
                    for perf in symbol_perfs:
                        symbol_data.append({
                            "銘柄": perf['symbol'],
                            "取引数": perf['total_trades'],
                            "勝率": f"{perf['win_rate']:.1f}%",
                            "総損益": f"${perf['total_pnl']:.2f}",
                            "平均損益": f"${perf['avg_pnl']:.2f}"
                        })
                    symbol_df = pd.DataFrame(symbol_data)
                    st.dataframe(symbol_df, use_container_width=True, hide_index=True)
                else:
                    st.info("銘柄別データなし")
            
            # レポート生成セクション
            st.subheader("📑 レポート生成")
            
            col_report1, col_report2, col_report3 = st.columns(3)
            
            with col_report1:
                if st.button("📄 HTMLレポート生成", help="ブラウザで表示可能な詳細レポート"):
                    with st.spinner("HTMLレポート生成中..."):
                        filepath = dashboard.trader.generate_performance_report('html')
                        if filepath:
                            st.success(f"✅ レポート生成完了")
                            st.write(f"保存先: {filepath}")
                            # ダウンロードリンク
                            with open(filepath, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.download_button(
                                label="📥 HTMLレポートをダウンロード",
                                data=html_content,
                                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
            
            with col_report2:
                if st.button("📊 PDFレポート生成", help="印刷用の正式レポート"):
                    with st.spinner("PDFレポート生成中..."):
                        filepath = dashboard.trader.generate_performance_report('pdf')
                        if filepath:
                            st.success(f"✅ レポート生成完了")
                            st.write(f"保存先: {filepath}")
                            # ダウンロードリンク
                            with open(filepath, 'rb') as f:
                                pdf_content = f.read()
                            st.download_button(
                                label="📥 PDFレポートをダウンロード",
                                data=pdf_content,
                                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
            
            with col_report3:
                if st.button("📋 取引履歴CSV出力", help="Excelで分析可能なデータ"):
                    with st.spinner("CSV出力中..."):
                        filepath = dashboard.trader.export_trade_history('csv')
                        if filepath:
                            st.success(f"✅ CSV出力完了")
                            st.write(f"保存先: {filepath}")
                            # ダウンロードリンク
                            with open(filepath, 'r', encoding='utf-8') as f:
                                csv_content = f.read()
                            st.download_button(
                                label="📥 CSVをダウンロード",
                                data=csv_content,
                                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
            
            # 予測精度分析
            prediction_accuracy = performance_summary.get('prediction_accuracy', {})
            if prediction_accuracy:
                st.subheader("🎯 予測精度分析")
                
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.metric(
                        "全体予測精度",
                        f"{prediction_accuracy.get('overall_accuracy', 0):.1%}",
                        f"総予測数: {prediction_accuracy.get('total_predictions', 0)}"
                    )
                
                with col_pred2:
                    # 信頼度別精度
                    confidence_acc = prediction_accuracy.get('confidence_accuracy', {})
                    if confidence_acc:
                        st.write("**信頼度別精度**")
                        for conf_range, data in confidence_acc.items():
                            st.write(f"{conf_range}: {data['accuracy']:.1%} ({data['count']}件)")
        else:
            st.info("パフォーマンスデータがありません。取引を実行してデータを蓄積してください。")
        
        # 詳細統計
        st.subheader("📈 詳細統計")
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.metric("総取引数", summary['performance_stats'].get('total_signals', 0))
            st.metric("成功取引", summary['performance_stats'].get('successful_trades', 0))
        
        with col_perf2:
            st.metric("失敗取引", summary['performance_stats'].get('failed_trades', 0))
            st.metric("最大連勝", summary['performance_stats'].get('max_consecutive_wins', 0))
        
        with col_perf3:
            st.metric("最大連敗", summary['performance_stats'].get('max_consecutive_losses', 0))
            current_streak = summary['performance_stats'].get('current_streak', 0)
            streak_type = summary['performance_stats'].get('streak_type', 'none')
            st.metric("現在の連続", f"{current_streak} ({streak_type})")
    
    with tab6:
        st.subheader("詳細設定")
        
        # リスク管理設定詳細
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.write("**ポジション管理**")
            st.write(f"1回あたりの取引サイズ: 資金の5%")
            st.write(f"最大同時ポジション数: 4")
            st.write(f"相関制限: 70%以上で減額")
            
            st.write("**ストップロス/テイクプロフィット**")
            st.write(f"基本ストップロス: 2%")
            st.write(f"基本テイクプロフィット: 4%")
            st.write(f"ボラティリティ調整: 有効")
        
        with col_set2:
            st.write("**リスク制限**")
            st.write(f"最大ポジションサイズ: {max_position}%")
            st.write(f"最大総エクスポージャー: {max_exposure}%")
            st.write(f"最大ドローダウン: {max_drawdown}%")
            
            st.write("**AI予測設定**")
            st.write(f"信頼度閾値: {confidence_threshold:.1%}")
            st.write(f"買いシグナル閾値: {buy_threshold:.1%}")
            st.write(f"売りシグナル閾値: {sell_threshold:.1%}")
        
        # システム情報
        st.subheader("システム情報")
        ml_available = summary.get('ml_performance', {}).get('ml_available', False)
        st.write("**予測エンジン**: 高度機械学習 + 従来手法アンサンブル")
        st.write("**ML利用状況**: " + ("✅ 利用可能" if ml_available else "❌ 基本モードのみ"))
        st.write("**リスク管理**: Kelly基準 + VaR + 相関分析")
        st.write("**データソース**: Hyperliquid DEX")
        st.write("**更新頻度**: 手動実行")
        
        # 予測精度目標
        st.subheader("📈 予測精度目標")
        st.write("**現在の目標**: 60%以上の予測精度")
        
        if prediction_stats:
            recent_accuracy = prediction_stats.get('average_confidence', 0)
            if recent_accuracy >= 0.6:
                st.success(f"🎯 目標達成！平均信頼度: {recent_accuracy:.1%}")
            elif recent_accuracy >= 0.5:
                st.warning(f"📊 改善中: 平均信頼度: {recent_accuracy:.1%}")
            else:
                st.error(f"🔄 要改善: 平均信頼度: {recent_accuracy:.1%}")
        
        # ML設定状況
        if ml_available:
            st.subheader("🤖 機械学習設定")
            training_samples = summary.get('ml_performance', {}).get('training_samples', 0)
            prediction_count = summary.get('ml_performance', {}).get('prediction_count', 0)
            
            st.write(f"**学習サンプル**: {training_samples}")
            st.write(f"**予測実行回数**: {prediction_count}")
            st.write(f"**再学習間隔**: 100回の予測毎")
            st.write(f"**次回再学習まで**: {100 - (prediction_count % 100)}回")
    
    with tab7:
        # マルチ銘柄管理タブ
        st.header("🌍 マルチ銘柄管理")
        
        # マルチ銘柄サマリー取得
        multi_summary = dashboard.trader.get_multi_symbol_summary()
        
        # 基本統計
        trading_summary = multi_summary.get('trading_summary', {})
        
        col_multi1, col_multi2, col_multi3, col_multi4 = st.columns(4)
        with col_multi1:
            st.metric("対応銘柄数", trading_summary.get('total_symbols', 0))
        with col_multi2:
            st.metric("有効銘柄数", trading_summary.get('enabled_symbols', 0))
        with col_multi3:
            st.metric("取引機会", trading_summary.get('trading_opportunities', 0))
        with col_multi4:
            st.metric("高信頼度シグナル", trading_summary.get('high_confidence_signals', 0))
        
        # 銘柄設定管理
        st.subheader("🎛️ 銘柄設定")
        
        col_symbols1, col_symbols2 = st.columns([2, 1])
        
        with col_symbols1:
            # 銘柄一覧表示
            all_symbols = dashboard.trader.multi_symbol_manager.get_all_symbols()
            enabled_symbols = dashboard.trader.multi_symbol_manager.get_enabled_symbols()
            
            symbol_data = []
            for symbol in all_symbols:
                config = dashboard.trader.multi_symbol_manager.get_symbol_config(symbol)
                if config:
                    symbol_data.append({
                        '銘柄': symbol,
                        '有効': '✅' if config.enabled else '❌',
                        '最大ポジション': f"{config.max_position_size:.1%}",
                        '最小信頼度': f"{config.min_confidence:.1%}",
                        'ストップロス': f"{config.stop_loss_pct:.1%}",
                        '利確': f"{config.take_profit_pct:.1%}"
                    })
            
            if symbol_data:
                df_symbols = pd.DataFrame(symbol_data)
                st.dataframe(df_symbols, use_container_width=True)
        
        with col_symbols2:
            # 銘柄管理コントロール
            st.subheader("銘柄操作")
            
            # 銘柄選択
            selected_symbol = st.selectbox("対象銘柄", all_symbols)
            
            # 有効/無効化
            col_toggle1, col_toggle2 = st.columns(2)
            with col_toggle1:
                if st.button("有効化", key=f"enable_{selected_symbol}"):
                    dashboard.trader.enable_symbol_trading(selected_symbol)
                    st.success(f"{selected_symbol} を有効化しました")
                    st.rerun()
            
            with col_toggle2:
                if st.button("無効化", key=f"disable_{selected_symbol}"):
                    dashboard.trader.disable_symbol_trading(selected_symbol)
                    st.success(f"{selected_symbol} を無効化しました")
                    st.rerun()
            
            # 設定調整
            st.subheader("詳細設定")
            config = dashboard.trader.multi_symbol_manager.get_symbol_config(selected_symbol)
            if config:
                new_max_pos = st.slider("最大ポジション", 0.01, 0.20, config.max_position_size, 0.01)
                new_min_conf = st.slider("最小信頼度", 0.50, 0.95, config.min_confidence, 0.05)
                new_stop = st.slider("ストップロス", 0.02, 0.15, config.stop_loss_pct, 0.01)
                new_profit = st.slider("利確", 0.05, 0.30, config.take_profit_pct, 0.01)
                
                if st.button("設定更新", key=f"update_{selected_symbol}"):
                    dashboard.trader.update_symbol_config(
                        selected_symbol,
                        max_position_size=new_max_pos,
                        min_confidence=new_min_conf,
                        stop_loss_pct=new_stop,
                        take_profit_pct=new_profit
                    )
                    st.success(f"{selected_symbol} の設定を更新しました")
                    st.rerun()
        
        # 取引機会分析
        opportunities = multi_summary.get('trading_opportunities', [])
        if opportunities:
            st.subheader("🎯 現在の取引機会")
            
            opp_data = []
            for opp in opportunities[:10]:  # 上位10件
                opp_data.append({
                    '銘柄': opp['symbol'],
                    'シグナル': opp['signal'],
                    '信頼度': f"{opp['confidence']:.1%}",
                    '価格': f"${opp['price']:,.2f}",
                    '最大ポジション': f"{opp['max_position_size']:.1%}",
                    'ストップロス': f"{opp['stop_loss_pct']:.1%}",
                    '利確': f"{opp['take_profit_pct']:.1%}"
                })
            
            if opp_data:
                df_opportunities = pd.DataFrame(opp_data)
                st.dataframe(df_opportunities, use_container_width=True)
        else:
            st.info("現在、取引機会はありません")
        
        # エクスポージャー分析
        exposure_analysis = multi_summary.get('exposure_analysis', {})
        if exposure_analysis:
            st.subheader("📊 ポートフォリオ分析")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                total_exp = exposure_analysis.get('total_exposure', 0)
                st.metric("総エクスポージャー", f"{total_exp:.1%}")
            
            with col_exp2:
                div_score = exposure_analysis.get('diversification_score', 0)
                st.metric("分散度スコア", f"{div_score:.1%}")
            
            with col_exp3:
                corr_risk = exposure_analysis.get('correlation_risk', 0)
                st.metric("相関リスク", f"{corr_risk:.1%}")
        
        # 実行ボタン
        st.subheader("⚡ マルチ銘柄取引実行")
        
        col_exec1, col_exec2, col_exec3 = st.columns(3)
        
        with col_exec1:
            if st.button("🚀 マルチ銘柄戦略実行", key="multi_execute"):
                with st.spinner("マルチ銘柄取引戦略を実行中..."):
                    dashboard.trader.execute_multi_symbol_strategy()
                st.success("マルチ銘柄戦略実行完了！")
                st.rerun()
        
        with col_exec2:
            if st.button("💾 設定保存", key="save_config"):
                dashboard.trader.save_multi_symbol_config()
                st.success("設定を保存しました")
        
        with col_exec3:
            if st.button("🔄 データ更新", key="refresh_multi"):
                st.rerun()
        
        # 銘柄別パフォーマンス
        symbol_metrics = multi_summary.get('symbol_metrics', {})
        if symbol_metrics:
            st.subheader("📈 銘柄別パフォーマンス")
            
            perf_data = []
            for symbol, metrics in symbol_metrics.items():
                perf_data.append({
                    '銘柄': symbol,
                    '現在価格': f"${metrics.get('current_price', 0):,.2f}",
                    '予測信頼度': f"{metrics.get('prediction_confidence', 0):.1%}",
                    'シグナル': metrics.get('prediction_signal', 'N/A'),
                    '最終更新': metrics.get('last_update', 'N/A')[:19] if metrics.get('last_update') else 'N/A'
                })
            
            if perf_data:
                df_performance = pd.DataFrame(perf_data)
                st.dataframe(df_performance, use_container_width=True)

if __name__ == "__main__":
    main()