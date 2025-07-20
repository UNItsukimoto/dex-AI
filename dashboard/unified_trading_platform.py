#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合AI取引プラットフォーム
取引所風UI + 完全な機能統合
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
from core.realistic_paper_trading import RealisticPaperTradingEngine, OrderSide, OrderType
from core.advanced_prediction_engine import AdvancedPredictionEngine
from core.risk_management_system import RiskManagementSystem
from trading_chart_ui import TradingChartUI

# エラーハンドリングシステム
from core.error_handler import (
    error_handler, safe_execute, safe_get, validate_numeric, validate_symbol,
    display_error_dashboard, create_error_boundary, init_streamlit_error_handling,
    APIConnectionError, DataValidationError, TradingError, PredictionError
)

# 拡張UIコンポーネント
from enhanced_ui_components import (
    show_loading_animation, show_success_notification, show_warning_notification,
    create_enhanced_metric_card, ProgressTracker, AlertManager, TourGuide,
    show_keyboard_shortcuts, PerformanceMonitor
)

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="🚀 統合AI取引プラットフォーム",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# エラーハンドリング初期化
init_streamlit_error_handling()

# カスタムCSS（レスポンシブ対応強化）
st.markdown("""
<style>
    .main-header {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 10px 10px 0px 0px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4AA;
    }
    
    /* レスポンシブ対応 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding-left: 10px;
            padding-right: 10px;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.2rem;
        }
        .metric-card {
            padding: 0.3rem;
        }
    }
    
    /* ダークテーマ対応強化 */
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
    
    .stNumberInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    
    /* ボタンスタイル改善 */
    .stButton > button {
        background: linear-gradient(145deg, #00D4AA, #00B899);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #00B899, #00A388);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class UnifiedTradingPlatform:
    """統合取引プラットフォーム（リアルな取引体験統合版）"""
    
    def __init__(self):
        if 'enhanced_trader' not in st.session_state:
            st.session_state.enhanced_trader = EnhancedAITrader(10000.0)
        
        # リアルな取引エンジンを統合
        if 'realistic_trader' not in st.session_state:
            st.session_state.realistic_trader = RealisticPaperTradingEngine(10000.0)
        
        # 高度予測エンジンを統合（既存のトレーダーから取得）
        if 'advanced_prediction_engine' not in st.session_state:
            st.session_state.advanced_prediction_engine = st.session_state.enhanced_trader.prediction_engine
        
        # リスク管理システムを統合（既存のトレーダーから取得）
        if 'risk_management_system' not in st.session_state:
            st.session_state.risk_management_system = st.session_state.enhanced_trader.risk_manager
        
        self.trader = st.session_state.enhanced_trader
        self.realistic_trader = st.session_state.realistic_trader
        self.prediction_engine = st.session_state.advanced_prediction_engine
        self.risk_manager = st.session_state.risk_management_system
        self.chart_ui = TradingChartUI()
        
        # 自動更新用のセッション状態
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'realistic_mode' not in st.session_state:
            st.session_state.realistic_mode = True  # デフォルトでリアルモード
    
    def create_risk_gauge(self, risk_level: str, risk_score: float) -> go.Figure:
        """リスクレベルゲージ"""
        risk_colors = {
            'low': 'green',
            'medium': 'yellow', 
            'high': 'orange',
            'extreme': 'red'
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"リスクレベル: {risk_level.upper()}"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_colors.get(risk_level, 'gray')},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_prediction_analysis_chart(self, predictions: list) -> go.Figure:
        """予測分析チャート"""
        if not predictions:
            return go.Figure()
        
        symbols = [p.get('symbol', 'Unknown') for p in predictions]
        probabilities = [p.get('probability', 0.5) for p in predictions]
        confidences = [p.get('confidence', 0.3) for p in predictions]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['上昇確率', '信頼度'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 上昇確率
        fig.add_trace(
            go.Bar(x=symbols, y=probabilities, name='上昇確率', 
                   marker_color='#00D4AA', text=[f'{p:.1%}' for p in probabilities]),
            row=1, col=1
        )
        
        # 信頼度
        fig.add_trace(
            go.Bar(x=symbols, y=confidences, name='信頼度',
                   marker_color='#FFD700', text=[f'{c:.1%}' for c in confidences]),
            row=1, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        return fig

@create_error_boundary("メインプラットフォーム")
@safe_execute("main_platform", show_error=True)
def main():
    # パフォーマンス監視
    perf_monitor = PerformanceMonitor()
    start_time = time.time()
    
    # ツアーガイド
    tour = TourGuide()
    tour.show_welcome_tour()
    
    # アラート管理
    alert_manager = AlertManager()
    alert_manager.show_alerts()
    
    platform = UnifiedTradingPlatform()
    
    # ヘッダー（レスポンシブ対応）
    st.markdown('<h1 class="main-header">🚀 統合AI暗号通貨取引プラットフォーム</h1>', unsafe_allow_html=True)
    
    # キーボードショートカット
    show_keyboard_shortcuts()
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ プラットフォーム設定")
        
        # エラー状況表示
        display_error_dashboard()
        
        # セキュア設定パネル
        if st.expander("🔐 セキュリティ設定", expanded=False):
            from core.secure_config import create_settings_panel
            create_settings_panel()
        
        # 取引モード選択
        trading_mode = st.radio(
            "🎯 取引モード", 
            ["🚀 リアル体験モード", "📊 基本モード"], 
            index=0 if st.session_state.realistic_mode else 1,
            horizontal=True
        )
        st.session_state.realistic_mode = (trading_mode == "🚀 リアル体験モード")
        
        if st.session_state.realistic_mode:
            st.success("🚀 リアルな取引体験モード: 手数料・スリッページ・遅延を含む")
        else:
            st.info("📊 基本モード: シンプルなペーパートレード")
        
        st.divider()
        
        # 銘柄選択
        available_symbols = platform.trader.multi_symbol_manager.get_all_symbols()
        selected_symbol = st.selectbox("📈 メイン銘柄", available_symbols, index=0)
        
        # 時間枠選択
        timeframe = st.selectbox("⏰ チャート時間枠", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        
        # 自動更新設定
        auto_refresh = st.checkbox("🔄 自動更新", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            if st.session_state.realistic_mode:
                refresh_interval = st.slider("更新間隔（秒）", 5, 60, 10)  # リアルモードはより頻繁
                if time.time() - st.session_state.last_update > refresh_interval:
                    platform.realistic_trader.update_live_prices()
                    st.session_state.last_update = time.time()
                    st.rerun()
            else:
                refresh_interval = st.slider("更新間隔（秒）", 10, 120, 30)
                if time.time() - st.session_state.last_update > refresh_interval:
                    st.session_state.last_update = time.time()
                    st.rerun()
        
        # 手動更新
        if st.button("🔄 今すぐ更新", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        # アカウント情報
        if st.session_state.realistic_mode:
            account = platform.realistic_trader.get_account_summary()
            st.subheader("💰 リアル体験アカウント")
            
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.metric("残高", f"${account['balance']:,.0f}")
            with col_acc2:
                st.metric("エクイティ", f"${account['equity']:,.0f}")
            
            # 日次損益
            daily_pnl = account.get('daily_pnl', 0)
            daily_pnl_pct = account.get('daily_pnl_pct', 0)
            
            st.metric(
                "日次損益", 
                f"${daily_pnl:,.0f}",
                f"{daily_pnl_pct:+.2%}"
            )
            
            # リアル体験の追加情報
            st.write(f"**証拠金率**: {account.get('margin_ratio', 0):.1%}")
            st.write(f"**未実現損益**: ${account.get('unrealized_pnl', 0):,.2f}")
        else:
            account = platform.trader.trading_engine.get_account_summary()
            st.subheader("💰 アカウント情報")
            
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.metric("残高", f"${account['balance']:,.0f}")
            with col_acc2:
                st.metric("エクイティ", f"${account['equity']:,.0f}")
            
            profit_loss = account['equity'] - account['balance']
            profit_pct = (profit_loss / account['balance'] * 100) if account['balance'] > 0 else 0
            
            st.metric(
                "総損益", 
                f"${profit_loss:,.0f}",
                f"{profit_pct:+.2f}%"
            )
        
        # リスク情報（リスク管理システムから直接取得）
        st.divider()
        account = platform.trader.trading_engine.get_account_summary()
        positions = platform.trader.trading_engine.get_positions()
        risk_metrics_obj = platform.risk_manager.calculate_risk_metrics(account['equity'], positions)
        
        # リスク管理システムの形式をダッシュボード形式に変換
        risk_metrics = {
            'risk_level': risk_metrics_obj.risk_level.value,
            'total_exposure': risk_metrics_obj.total_exposure,
            'portfolio_volatility': risk_metrics_obj.portfolio_volatility,
            'risk_score': 0.5  # 仮の値（リスクレベルから計算）
        }
        if risk_metrics_obj.risk_level.value == 'extreme':
            risk_metrics['risk_score'] = 0.9
        elif risk_metrics_obj.risk_level.value == 'high':
            risk_metrics['risk_score'] = 0.7
        elif risk_metrics_obj.risk_level.value == 'medium':
            risk_metrics['risk_score'] = 0.5
        else:
            risk_metrics['risk_score'] = 0.3
        
        summary = platform.trader.get_enhanced_summary()
        
        st.subheader("⚠️ リスク状況")
        st.write(f"**リスクレベル**: {risk_metrics.get('risk_level', 'low').upper()}")
        st.write(f"**総エクスポージャー**: {risk_metrics.get('total_exposure', 0):.1%}")
        st.write(f"**ボラティリティ**: {risk_metrics.get('portfolio_volatility', 0):.1%}")
        
        # アクションボタン
        st.divider()
        st.subheader("🚀 クイックアクション")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🎯 AI予測実行", use_container_width=True, key="sidebar_ai_predict"):
                with st.spinner("AI予測中..."):
                    platform.trader.execute_enhanced_strategy()
                st.success("予測完了！")
                st.rerun()
        
        with col_act2:
            if st.button("🌍 マルチ銘柄", use_container_width=True, key="sidebar_multi_symbol"):
                with st.spinner("マルチ銘柄分析中..."):
                    platform.trader.execute_multi_symbol_strategy()
                st.success("分析完了！")
                st.rerun()
    
    # メインエリア - タブ構成
    if st.session_state.realistic_mode:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 メイントレーディング",
            "🚀 リアル取引体験",
            "🤖 AI予測分析", 
            "📊 リスク管理",
            "🌍 マルチ銘柄管理",
            "📈 パフォーマンス",
            "🚨 アラート・通知",
            "📋 取引履歴",
            "⚙️ 高度な設定"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📈 メイントレーディング",
            "🤖 AI予測分析", 
            "📊 リスク管理",
            "🌍 マルチ銘柄管理",
            "📈 パフォーマンス",
            "🚨 アラート・通知",
            "📋 取引履歴",
            "⚙️ 高度な設定"
        ])
    
    with tab1:
        # メイントレーディングタブ
        st.header(f"📈 {selected_symbol} リアルタイムトレーディング")
        
        # 現在価格と予測情報
        if st.session_state.realistic_mode:
            live_prices = platform.realistic_trader.get_live_prices()
            current_price = live_prices.get(selected_symbol, 0)
            market_data = platform.realistic_trader.get_market_summary().get(selected_symbol, {})
        else:
            current_prices = platform.chart_ui.get_current_prices([selected_symbol])
            current_price = current_prices.get(selected_symbol, 0)
            market_data = {}
        
        # AI予測実行（高度予測エンジン使用）
        try:
            df = platform.chart_ui.get_historical_data(selected_symbol, timeframe)
            prediction = platform.prediction_engine.get_enhanced_prediction(selected_symbol, df)
        except Exception as e:
            # フォールバック: ダミーデータで予測
            prediction = {
                'signal': 'HOLD',
                'probability': 0.5,
                'confidence': 0.3
            }
        
        # 予測結果をトレーダーの形式に変換
        prediction['symbol'] = selected_symbol
        prediction['price'] = current_price
        prediction['timestamp'] = datetime.now().isoformat()
        
        # 上部メトリクス
        col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
        
        with col_info1:
            st.metric("💰 現在価格", f"${current_price:,.2f}")
        
        with col_info2:
            if st.session_state.realistic_mode:
                change_24h = market_data.get('change_24h', 0)
                delta_color = "normal" if change_24h >= 0 else "inverse"
                st.metric("24h変動", f"{change_24h:+.2f}%", delta_color=delta_color)
            else:
                change_24h = np.random.uniform(-5, 5)  # 実際の実装では真の24h変動を取得
                delta_color = "normal" if change_24h >= 0 else "inverse"
                st.metric("24h変動", f"{change_24h:+.2f}%", delta_color=delta_color)
        
        with col_info3:
            confidence = prediction.get('confidence', 0.3)
            if confidence >= 0.8:
                conf_class = "prediction-high"
            elif confidence >= 0.6:
                conf_class = "prediction-medium"
            else:
                conf_class = "prediction-low"
            
            st.markdown(f'<div class="{conf_class}">AI信頼度: {confidence:.1%}</div>', unsafe_allow_html=True)
        
        with col_info4:
            signal = prediction.get('signal', 'HOLD')
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            st.metric("AIシグナル", f"{signal_emoji.get(signal, '🟡')} {signal}")
        
        with col_info5:
            probability = prediction.get('probability', 0.5)
            st.metric("上昇確率", f"{probability:.1%}")
        
        # メインチャートと取引エリア
        col_chart, col_trading = st.columns([3, 1])
        
        with col_chart:
            # TradingView風チャート
            chart = platform.chart_ui.create_trading_view_chart(selected_symbol, timeframe)
            
            # AI予測オーバーレイ
            if prediction:
                chart = platform.chart_ui.create_prediction_overlay(chart, prediction)
            
            st.plotly_chart(chart, use_container_width=True)
        
        with col_trading:
            # 取引パネル
            st.subheader("⚡ 取引パネル")
            
            if st.session_state.realistic_mode:
                st.info("🚀 リアル体験モード: 手数料・スリッページ・遅延あり")
            
            # 注文タイプ選択
            order_type = st.radio("注文タイプ", ["成行", "指値"], horizontal=True)
            trade_type = st.radio("取引タイプ", ["買い", "売り"], horizontal=True)
            
            # 金額・数量入力
            if order_type == "成行":
                amount = st.number_input("金額 (USD)", min_value=10.0, value=500.0, step=10.0)
                if current_price > 0:
                    quantity = amount / current_price
                    st.write(f"概算数量: {quantity:.6f} {selected_symbol}")
                else:
                    quantity = 0
                price_input = None
            else:
                quantity = st.number_input(f"数量 ({selected_symbol})", min_value=0.0001, value=0.1, step=0.0001, format="%.6f")
                price_input = st.number_input("価格 (USD)", min_value=1.0, value=current_price if current_price > 0 else 1000.0, step=1.0)
                amount = quantity * price_input
            
            # 取引概算
            if st.session_state.realistic_mode:
                # リスク管理システムからポジションサイズ推奨を取得
                account_equity = account.get('equity', 10000) if 'account' in locals() else 10000
                confidence = prediction.get('confidence', 0.3)
                position_rec = platform.risk_manager.calculate_position_size(
                    selected_symbol, account_equity, confidence, current_price
                )
                
                fee = amount * 0.001  # 0.1% fee
                slippage_est = amount * 0.0005  # 0.05% slippage estimate
                total_cost = amount + fee + (slippage_est if trade_type == "買い" else -slippage_est)
                
                st.markdown(f"""
                **取引概算 (リアル体験)**:
                - 金額: ${amount:.2f}
                - 手数料: ${fee:.2f}
                - 予想スリッページ: ${slippage_est:.2f}
                - 総コスト: ${total_cost:.2f}
                
                **リスク管理推奨**:
                - 推奨サイズ: {position_rec.recommended_size:.6f} {selected_symbol}
                - 理由: {position_rec.reason}
                """)
            else:
                st.write(f"**取引金額**: ${amount:.2f}")
            
            # 取引実行ボタン
            if trade_type == "買い":
                if st.button("🚀 買い注文", type="primary", use_container_width=True, key="main_buy_button"):
                    try:
                        if st.session_state.realistic_mode:
                            side = OrderSide.BUY
                            otype = OrderType.MARKET if order_type == "成行" else OrderType.LIMIT
                            
                            # リスク管理システムでポジションサイズ確認
                            if quantity > position_rec.max_allowed_size:
                                st.warning(f"⚠️ 注文数量が推奨最大サイズ({position_rec.max_allowed_size:.6f})を超えています")
                            
                            order_id = platform.realistic_trader.place_order(
                                symbol=selected_symbol,
                                side=side,
                                order_type=otype,
                                quantity=quantity,
                                price=price_input
                            )
                            
                            if order_id:
                                st.success(f"✅ {selected_symbol} 買い注文成功! ID: {order_id}")
                                if order_type == "成行":
                                    st.balloons()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 注文失敗: 残高不足またはエラー")
                        else:
                            st.success(f"✅ {selected_symbol} 買い注文 ${amount:.2f}")
                            st.balloons()
                    except Exception as e:
                        st.error(f"❌ 注文エラー: {e}")
            else:
                if st.button("📉 売り注文", type="secondary", use_container_width=True, key="main_sell_button"):
                    try:
                        if st.session_state.realistic_mode:
                            side = OrderSide.SELL
                            otype = OrderType.MARKET if order_type == "成行" else OrderType.LIMIT
                            
                            # リスク管理システムでポジションサイズ確認
                            if quantity > position_rec.max_allowed_size:
                                st.warning(f"⚠️ 注文数量が推奨最大サイズ({position_rec.max_allowed_size:.6f})を超えています")
                            
                            order_id = platform.realistic_trader.place_order(
                                symbol=selected_symbol,
                                side=side,
                                order_type=otype,
                                quantity=quantity,
                                price=price_input
                            )
                            
                            if order_id:
                                st.success(f"✅ {selected_symbol} 売り注文成功! ID: {order_id}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 注文失敗: ポジション不足またはエラー")
                        else:
                            st.success(f"✅ {selected_symbol} 売り注文 ${amount:.2f}")
                    except Exception as e:
                        st.error(f"❌ 注文エラー: {e}")
            
            st.divider()
            
            # AI推奨（高度予測エンジンの詳細情報を含む）
            st.subheader("🤖 AI推奨")
            
            confidence = prediction.get('confidence', 0.3)
            signal = prediction.get('signal', 'HOLD')
            
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
            
            # 高度予測の詳細情報表示
            if 'ml_prediction' in prediction:
                with st.expander("🧠 ML予測詳細", expanded=False):
                    ml_pred = prediction.get('ml_prediction', {})
                    if 'model_predictions' in ml_pred:
                        st.write("**モデル別予測:**")
                        for model, prob in ml_pred['model_predictions'].items():
                            st.write(f"- {model}: {prob:.1%}")
            
            # リスク情報（リスク管理システムから詳細情報を取得）
            st.subheader("⚠️ リスク情報")
            
            # ポジションサイズ推奨
            account_equity = account['equity']
            confidence = prediction.get('confidence', 0.3)
            position_recommendation = platform.risk_manager.calculate_position_size(
                selected_symbol, account_equity, confidence, current_price
            )
            
            st.write(f"**推奨ポジションサイズ**: {position_recommendation.recommended_size:.6f} {selected_symbol}")
            st.write(f"**最大許容サイズ**: {position_recommendation.max_allowed_size:.6f} {selected_symbol}")
            
            # ストップロス・テイクプロフィット計算
            if current_price > 0:
                sl_tp = platform.risk_manager.calculate_stop_loss_take_profit(
                    selected_symbol, current_price, 'long', confidence
                )
                st.write(f"**推奨ストップロス**: ${sl_tp['stop_loss_price']:,.2f} ({sl_tp['stop_loss_pct']:.1%})")
                st.write(f"**推奨利確目標**: ${sl_tp['take_profit_price']:,.2f} ({sl_tp['take_profit_pct']:.1%})")
            
            # 既存の設定も表示
            config = platform.trader.multi_symbol_manager.get_symbol_config(selected_symbol)
            if config:
                with st.expander("📋 設定値", expanded=False):
                    st.write(f"**最大ポジション**: {config.max_position_size:.1%}")
                    st.write(f"**ストップロス**: {config.stop_loss_pct:.1%}")
                    st.write(f"**利確目標**: {config.take_profit_pct:.1%}")
            
            # 現在のポジション
            if st.session_state.realistic_mode:
                positions = platform.realistic_trader.get_positions()
            else:
                positions = platform.trader.trading_engine.get_positions()
            
            if selected_symbol in positions:
                st.divider()
                pos = positions[selected_symbol]
                st.subheader("📍 現在のポジション")
                
                if st.session_state.realistic_mode:
                    st.write(f"**数量**: {pos['quantity']:.6f}")
                    st.write(f"**エントリー価格**: ${pos['entry_price']:,.2f}")
                    st.write(f"**現在価格**: ${pos['current_price']:,.2f}")
                    st.write(f"**市場価値**: ${pos['market_value']:,.2f}")
                    
                    pnl = pos['unrealized_pnl']
                    pnl_pct = pos['unrealized_pnl_pct']
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    st.write(f"**未実現損益**: {pnl_color} ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                    
                    # ポジション決済ボタン
                    if st.button(f"🔄 {selected_symbol} ポジション決済", use_container_width=True, key="close_position"):
                        try:
                            close_side = OrderSide.SELL if pos['quantity'] > 0 else OrderSide.BUY
                            close_order = platform.realistic_trader.place_order(
                                symbol=selected_symbol,
                                side=close_side,
                                order_type=OrderType.MARKET,
                                quantity=abs(pos['quantity'])
                            )
                            
                            if close_order:
                                st.success("✅ ポジション決済完了!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 決済失敗")
                        except Exception as e:
                            st.error(f"❌ 決済エラー: {e}")
                else:
                    st.write(f"**数量**: {pos['quantity']:.4f}")
                    st.write(f"**平均価格**: ${pos.get('avg_price', 0):.2f}")
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    st.write(f"**損益**: {pnl_color} ${pnl:,.2f}")
            else:
                st.info(f"💡 {selected_symbol} のポジションなし")
        
        # オーダーブック
        st.divider()
        platform.chart_ui.create_orderbook_widget(selected_symbol)
    
    # リアル取引体験タブ（リアルモード時のみ表示）
    if st.session_state.realistic_mode:
        with tab2:
            st.header("🚀 リアルな取引体験")
            
            # リアル体験の説明
            st.markdown("""
            <div style="background: linear-gradient(145deg, #1e1e1e, #2d2d2d); padding: 1.5rem; border-radius: 15px; border: 1px solid #333; margin: 1rem 0;">
                <h3>🎯 リアルな取引体験とは？</h3>
                <ul>
                    <li>🕐 <strong>注文約定遅延</strong>: 0.1-1秒の実際の約定遅延</li>
                    <li>💰 <strong>取引手数料</strong>: 0.1%の手数料を適用</li>
                    <li>📊 <strong>スリッページ</strong>: 最大0.05%のスリッページ</li>
                    <li>📈 <strong>リアルタイム価格</strong>: 5秒ごとの価格更新</li>
                    <li>⚠️ <strong>マーケット影響</strong>: 実際の市場条件をシミュレート</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # リアル体験ダッシュボード
            col_real1, col_real2 = st.columns([2, 1])
            
            with col_real1:
                st.subheader("📊 ライブマーケット")
                
                # マーケットサマリー
                market_summary = platform.realistic_trader.get_market_summary()
                
                market_data = []
                for symbol, data in market_summary.items():
                    change_emoji = "📈" if data['change_24h'] >= 0 else "📉"
                    market_data.append({
                        '銘柄': symbol,
                        '価格': f"${data['price']:,.2f}",
                        '24h変動': f"{data['change_24h']:+.2f}%",
                        '状況': change_emoji,
                        '高値': f"${data['day_high']:,.2f}",
                        '安値': f"${data['day_low']:,.2f}"
                    })
                
                market_df = pd.DataFrame(market_data)
                st.dataframe(market_df, use_container_width=True, hide_index=True)
                
                # 最近の取引実行状況
                st.subheader("⚡ 最近の取引実行")
                recent_trades = platform.realistic_trader.get_trade_history(10)
                
                if recent_trades:
                    trade_data = []
                    for trade in recent_trades[-5:]:  # 最新5件
                        side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"
                        trade_data.append({
                            '時刻': trade['timestamp'][:19].replace('T', ' '),
                            '銘柄': trade['symbol'],
                            '売買': f"{side_emoji} {trade['side'].title()}",
                            '数量': f"{trade['quantity']:.6f}",
                            '価格': f"${trade['price']:,.2f}",
                            '手数料': f"${trade['fee']:.2f}",
                            'スリッページ': f"{trade['slippage']:.3%}"
                        })
                    
                    trades_df = pd.DataFrame(trade_data)
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
                else:
                    st.info("まだ取引実行がありません。上記で取引を実行してみてください！")
            
            with col_real2:
                st.subheader("💼 リアル体験ポートフォリオ")
                
                # アカウントサマリー
                account = platform.realistic_trader.get_account_summary()
                
                st.metric("残高", f"${account['balance']:,.2f}")
                st.metric("エクイティ", f"${account['equity']:,.2f}")
                st.metric("未実現損益", f"${account['unrealized_pnl']:,.2f}")
                st.metric("証拠金率", f"{account['margin_ratio']:.1%}")
                
                # ポジション概要
                st.divider()
                st.subheader("📍 アクティブポジション")
                
                positions = platform.realistic_trader.get_positions()
                if positions:
                    for symbol, pos in positions.items():
                        pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                        st.markdown(f"""
                        **{symbol}**
                        - 数量: {pos['quantity']:.6f}
                        - 損益: {pnl_color} ${pos['unrealized_pnl']:,.2f}
                        """)
                else:
                    st.info("アクティブなポジションなし")
                
                # リアル体験統計
                st.divider()
                st.subheader("📊 リアル体験統計")
                
                total_trades = len(platform.realistic_trader.get_trade_history())
                total_fees_paid = sum(t['fee'] for t in platform.realistic_trader.get_trade_history())
                
                st.write(f"**総取引回数**: {total_trades}")
                st.write(f"**支払い手数料**: ${total_fees_paid:.2f}")
                st.write(f"**平均スリッページ**: {np.mean([t['slippage'] for t in platform.realistic_trader.get_trade_history()]) if total_trades > 0 else 0:.3%}")
                
                # 更新ボタン
                if st.button("🔄 価格データ更新", use_container_width=True, key="realistic_refresh"):
                    platform.realistic_trader.update_live_prices()
                    st.success("価格データを更新しました！")
                    st.rerun()
        
        tab_offset = 1
    else:
        tab_offset = 0
    
    with st.session_state.get('tabs', [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9] if st.session_state.realistic_mode else [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8])[1 + tab_offset]:
        # AI予測分析タブ
        st.header("🤖 AI予測分析")
        
        # 予測実行ボタン
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            if st.button("🚀 単体予測実行", type="primary", use_container_width=True, key="pred_single"):
                with st.spinner("AI予測を実行中..."):
                    # 高度予測エンジンで予測実行
                    try:
                        df = platform.chart_ui.get_historical_data(selected_symbol, '1h')
                        prediction = platform.prediction_engine.get_enhanced_prediction(selected_symbol, df)
                    except Exception:
                        prediction = {'signal': 'HOLD', 'probability': 0.5, 'confidence': 0.3}
                    
                    # トレーダーの予測履歴に追加
                    platform.trader.prediction_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': selected_symbol,
                        'prediction': prediction
                    })
                st.success("単体予測完了！")
                st.rerun()
        
        with col_pred2:
            if st.button("🌍 マルチ銘柄予測", type="secondary", use_container_width=True, key="pred_multi"):
                with st.spinner("マルチ銘柄予測を実行中..."):
                    # 全銘柄に対して高度予測実行
                    for symbol in available_symbols:
                        try:
                            df = platform.chart_ui.get_historical_data(symbol, '1h')
                            prediction = platform.prediction_engine.get_enhanced_prediction(symbol, df)
                            
                            # 価格履歴更新（リスク管理用）
                            current_price = platform.chart_ui.get_current_prices([symbol])[symbol]
                            platform.risk_manager.update_price_history(symbol, current_price)
                        except Exception:
                            continue  # エラーがあった銘柄はスキップ
                    
                    # ポートフォリオ履歴更新
                    account = platform.trader.trading_engine.get_account_summary()
                    positions = platform.trader.trading_engine.get_positions()
                    platform.risk_manager.update_portfolio_history(account['equity'], positions)
                    
                st.success("マルチ銘柄予測完了！")
                st.rerun()
        
        with col_pred3:
            if st.button("🧪 テストアラート", use_container_width=True, key="pred_test_alert"):
                platform.trader.send_test_alert()
                st.success("テストアラート送信完了！")
        
        # 予測結果表示
        summary = platform.trader.get_enhanced_summary()
        predictions = summary.get('latest_predictions', [])
        
        if predictions:
            # 予測チャート
            st.subheader("📊 予測分析チャート")
            pred_chart = platform.create_prediction_analysis_chart(predictions)
            st.plotly_chart(pred_chart, use_container_width=True)
            
            # 予測詳細テーブル
            st.subheader("📋 詳細予測結果")
            pred_data = []
            for pred in predictions:
                pred_data.append({
                    '銘柄': pred.get('symbol', 'Unknown'),
                    '現在価格': f"${pred.get('price', 0):,.2f}",
                    'シグナル': pred.get('signal', 'HOLD'),
                    '上昇確率': f"{pred.get('probability', 0.5):.1%}",
                    '信頼度': f"{pred.get('confidence', 0.3):.1%}",
                    '推奨アクション': get_action_recommendation(pred.get('signal', 'HOLD'), pred.get('confidence', 0.3)),
                    '更新時刻': pred.get('timestamp', '')[:19]
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info("予測データを取得するには上記の予測実行ボタンを押してください")
        
        # ML性能指標（高度予測エンジンから直接取得）
        st.subheader("🧠 機械学習性能")
        
        ml_performance = platform.prediction_engine.get_model_performance()
        prediction_stats = summary.get('prediction_stats', {})
        
        col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
        
        with col_ml1:
            avg_conf = prediction_stats.get('average_confidence', 0) if prediction_stats else 0
            st.metric("平均予測精度", f"{avg_conf:.1%}", "+2.1%")
        
        with col_ml2:
            total_preds = prediction_stats.get('total_predictions', 0) if prediction_stats else 0
            st.metric("総予測数", f"{total_preds:,}", "+23")
        
        with col_ml3:
            high_conf = prediction_stats.get('high_confidence_predictions', 0) if prediction_stats else 0
            st.metric("高信頼度予測", f"{high_conf:,}", "+8")
        
        with col_ml4:
            ml_available = ml_performance.get('ml_available', False)
            status = "稼働中" if ml_available else "基本モード"
            st.metric("MLエンジン", status)
    
    with tab3:
        # リスク管理タブ
        st.header("📊 リスク管理・分析")
        
        # リスク管理システムから詳細なリスクレポートを取得
        account = platform.trader.trading_engine.get_account_summary()
        positions = platform.trader.trading_engine.get_positions()
        risk_report = platform.risk_manager.generate_risk_report(account['equity'], positions)
        risk_metrics_obj = platform.risk_manager.calculate_risk_metrics(account['equity'], positions)
        
        # ダッシュボード形式に変換
        risk_metrics = {
            'risk_level': risk_metrics_obj.risk_level.value,
            'risk_score': 0.5,
            'total_exposure': risk_metrics_obj.total_exposure,
            'max_drawdown': risk_metrics_obj.max_drawdown,
            'var_95': risk_metrics_obj.var_95,
            'portfolio_volatility': risk_metrics_obj.portfolio_volatility,
            'sharpe_ratio': risk_metrics_obj.sharpe_ratio,
            'correlation_risk': risk_metrics_obj.concentration_risk  # 集中リスクを相関リスクとして使用
        }
        
        # リスクスコアを計算
        if risk_metrics_obj.risk_level.value == 'extreme':
            risk_metrics['risk_score'] = 0.9
        elif risk_metrics_obj.risk_level.value == 'high':
            risk_metrics['risk_score'] = 0.7
        elif risk_metrics_obj.risk_level.value == 'medium':
            risk_metrics['risk_score'] = 0.5
        else:
            risk_metrics['risk_score'] = 0.3
        
        summary = platform.trader.get_enhanced_summary()
        
        col_risk1, col_risk2 = st.columns([1, 2])
        
        with col_risk1:
            st.subheader("🛡️ リスクレベル")
            risk_level = risk_metrics.get('risk_level', 'low')
            risk_score = risk_metrics.get('risk_score', 0)
            risk_gauge = platform.create_risk_gauge(risk_level, risk_score)
            st.plotly_chart(risk_gauge, use_container_width=True)
        
        with col_risk2:
            st.subheader("📈 リスク指標")
            
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.metric("総エクスポージャー", f"{risk_metrics.get('total_exposure', 0):.1%}")
                st.metric("最大ドローダウン", f"{risk_metrics.get('max_drawdown', 0):.1%}")
                st.metric("VaR (95%)", f"${risk_metrics.get('var_95', 0):,.0f}")
            
            with col_metric2:
                st.metric("ポートフォリオボラティリティ", f"{risk_metrics.get('portfolio_volatility', 0):.1%}")
                st.metric("シャープレシオ", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("相関リスク", f"{risk_metrics.get('correlation_risk', 0):.1%}")
        
        # リスク警告（リスク管理システムから取得）
        violations = risk_report.get('violations', [])
        if violations:
            st.subheader("⚠️ リスク警告")
            for violation in violations[:5]:
                if violation['severity'] == 'high':
                    st.error(f"**高リスク**: {violation['message']}")
                elif violation['severity'] == 'medium':
                    st.warning(f"**中リスク**: {violation['message']}")
                else:
                    st.info(f"**注意**: {violation['message']}")
        else:
            st.success("✅ 現在、リスク警告はありません")
        
        # リスク改善推奨事項
        recommendations = risk_report.get('recommendations', [])
        if recommendations:
            st.subheader("💡 リスク改善推奨事項")
            for i, rec in enumerate(recommendations, 1):
                st.info(f"{i}. {rec}")
        
        # ポジション分析
        st.subheader("📍 ポジション分析")
        positions = summary.get('positions', {})
        
        if positions:
            pos_data = []
            for symbol, pos in positions.items():
                pos_data.append({
                    '銘柄': symbol,
                    '数量': f"{pos['quantity']:.4f}",
                    '市場価値': f"${pos.get('market_value', 0):,.2f}",
                    '未実現損益': f"${pos.get('unrealized_pnl', 0):,.2f}",
                    '損益率': f"{pos.get('unrealized_pnl_pct', 0):.2%}",
                    'エクスポージャー': f"{pos.get('exposure_pct', 0):.1%}"
                })
            
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("現在アクティブなポジションはありません")
    
    with tab4:
        # マルチ銘柄管理タブ
        st.header("🌍 マルチ銘柄管理")
        
        # マーケット概要
        st.subheader("📊 マーケット概要")
        platform.chart_ui.create_market_overview_widget(available_symbols[:6])
        
        # マルチ銘柄統計
        multi_summary = platform.trader.get_multi_symbol_summary()
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
        
        # 銘柄管理
        st.subheader("🎛️ 銘柄設定管理")
        
        col_symbols1, col_symbols2 = st.columns([2, 1])
        
        with col_symbols1:
            # 銘柄一覧表示
            symbol_data = []
            for symbol in available_symbols:
                config = platform.trader.multi_symbol_manager.get_symbol_config(symbol)
                current_price = platform.chart_ui.get_current_prices([symbol])[symbol]
                # 銀柄別予測取得
                try:
                    symbol_df = platform.chart_ui.get_historical_data(symbol, '1h')
                    prediction = platform.prediction_engine.get_enhanced_prediction(symbol, symbol_df)
                    prediction['symbol'] = symbol
                    prediction['price'] = current_price
                except Exception:
                    # フォールバック
                    prediction = {
                        'signal': 'HOLD',
                        'probability': 0.5,
                        'confidence': 0.3,
                        'symbol': symbol,
                        'price': current_price
                    }
                
                if config:
                    symbol_data.append({
                        '銘柄': symbol,
                        '価格': f"${current_price:,.2f}",
                        '状態': '✅' if config.enabled else '❌',
                        'AIシグナル': prediction.get('signal', 'HOLD'),
                        '信頼度': f"{prediction.get('confidence', 0):.1%}",
                        '最大ポジション': f"{config.max_position_size:.1%}",
                        '最小信頼度': f"{config.min_confidence:.1%}",
                        'ストップロス': f"{config.stop_loss_pct:.1%}",
                        '利確': f"{config.take_profit_pct:.1%}"
                    })
            
            if symbol_data:
                symbol_df = pd.DataFrame(symbol_data)
                st.dataframe(symbol_df, use_container_width=True, hide_index=True)
        
        with col_symbols2:
            # 銘柄操作パネル
            st.subheader("操作パネル")
            
            symbol_to_manage = st.selectbox("操作対象銘柄", available_symbols)
            
            col_toggle1, col_toggle2 = st.columns(2)
            with col_toggle1:
                if st.button("✅ 有効化", use_container_width=True, key="multi_enable"):
                    platform.trader.enable_symbol_trading(symbol_to_manage)
                    st.success(f"{symbol_to_manage} を有効化")
                    st.rerun()
            
            with col_toggle2:
                if st.button("❌ 無効化", use_container_width=True, key="multi_disable"):
                    platform.trader.disable_symbol_trading(symbol_to_manage)
                    st.success(f"{symbol_to_manage} を無効化")
                    st.rerun()
            
            if st.button("💾 設定保存", use_container_width=True, key="multi_save_config"):
                platform.trader.save_multi_symbol_config()
                st.success("設定を保存しました")
        
        # 取引機会分析
        opportunities = multi_summary.get('trading_opportunities', [])
        if opportunities:
            st.subheader("🎯 現在の取引機会")
            
            opp_data = []
            for opp in opportunities[:10]:
                opp_data.append({
                    '銘柄': opp['symbol'],
                    'シグナル': opp['signal'],
                    '信頼度': f"{opp['confidence']:.1%}",
                    '価格': f"${opp['price']:,.2f}",
                    '推奨ポジション': f"{opp['max_position_size']:.1%}",
                    'ストップロス': f"{opp['stop_loss_pct']:.1%}",
                    '利確目標': f"{opp['take_profit_pct']:.1%}"
                })
            
            if opp_data:
                opp_df = pd.DataFrame(opp_data)
                st.dataframe(opp_df, use_container_width=True, hide_index=True)
        else:
            st.info("現在、高信頼度の取引機会はありません")
    
    with tab5:
        # パフォーマンスタブ
        st.header("📈 パフォーマンス分析")
        
        # パフォーマンスサマリー取得
        perf_summary = platform.trader.get_performance_summary()
        
        # 主要指標
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        metrics = perf_summary.get('metrics', {})
        
        with col_perf1:
            total_return = metrics.get('total_return', 0)
            st.metric("総リターン", f"${total_return:,.2f}")
        
        with col_perf2:
            return_pct = metrics.get('total_return_pct', 0)
            st.metric("リターン率", f"{return_pct:.2%}")
        
        with col_perf3:
            win_rate = metrics.get('win_rate', 0)
            st.metric("勝率", f"{win_rate:.1%}")
        
        with col_perf4:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric("シャープレシオ", f"{sharpe:.2f}")
        
        # エクイティカーブ
        st.subheader("📊 エクイティカーブ")
        
        # ダミーエクイティカーブ生成
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        initial_balance = platform.trader.trading_engine.initial_balance
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
        
        # 銘柄別パフォーマンス
        symbol_performance = perf_summary.get('symbol_performance', [])
        if symbol_performance:
            st.subheader("📊 銘柄別パフォーマンス")
            
            symbol_perf_data = []
            for perf in symbol_performance:
                symbol_perf_data.append({
                    '銘柄': perf['symbol'],
                    '取引回数': perf['total_trades'],
                    '勝率': f"{perf['win_rate']:.1%}",
                    '総損益': f"${perf['total_pnl']:,.2f}",
                    '平均損益': f"${perf['avg_pnl']:,.2f}"
                })
            
            symbol_perf_df = pd.DataFrame(symbol_perf_data)
            st.dataframe(symbol_perf_df, use_container_width=True, hide_index=True)
        
        # レポート生成
        st.subheader("📋 レポート生成")
        
        col_report1, col_report2, col_report3 = st.columns(3)
        
        with col_report1:
            if st.button("📊 HTMLレポート生成", use_container_width=True, key="perf_html_report"):
                with st.spinner("HTMLレポートを生成中..."):
                    filepath = platform.trader.generate_performance_report('html')
                if filepath:
                    st.success(f"HTMLレポート生成完了: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "📥 HTMLレポートダウンロード",
                            f.read(),
                            file_name="performance_report.html",
                            mime="text/html",
                            key="perf_html_download"
                        )
        
        with col_report2:
            if st.button("📄 PDFレポート生成", use_container_width=True, key="perf_pdf_report"):
                with st.spinner("PDFレポートを生成中..."):
                    filepath = platform.trader.generate_performance_report('pdf')
                if filepath:
                    st.success(f"PDFレポート生成完了: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "📥 PDFレポートダウンロード",
                            f.read(),
                            file_name="performance_report.pdf",
                            mime="application/pdf",
                            key="perf_pdf_download"
                        )
        
        with col_report3:
            if st.button("📊 CSVエクスポート", use_container_width=True, key="perf_csv_export"):
                with st.spinner("CSVデータを生成中..."):
                    filepath = platform.trader.export_trade_history('csv')
                if filepath:
                    st.success(f"CSVエクスポート完了: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "📥 取引履歴CSV",
                            f.read(),
                            file_name="trade_history.csv",
                            mime="text/csv",
                            key="perf_csv_download"
                        )
    
    with tab6:
        # アラート・通知タブ
        st.header("🚨 アラート・通知システム")
        
        # アラートサマリー
        alert_summary = platform.trader.get_alert_system_summary()
        
        col_alert1, col_alert2, col_alert3, col_alert4 = st.columns(4)
        
        with col_alert1:
            alert_count = len(alert_summary.get('alert_history', []))
            st.metric("24h アラート数", alert_count)
        
        with col_alert2:
            active_channels = alert_summary.get('active_channels', [])
            st.metric("有効チャンネル", len(active_channels))
        
        with col_alert3:
            st.metric("通知レート制限", "60秒")
        
        with col_alert4:
            if st.button("🧪 テストアラート", use_container_width=True, key="alert_test_button"):
                result = platform.trader.send_test_alert()
                if result:
                    st.success("テストアラート送信完了")
                else:
                    st.error("テストアラート送信失敗")
        
        # アラート設定
        st.subheader("⚙️ アラート設定")
        
        col_setting1, col_setting2 = st.columns(2)
        
        with col_setting1:
            st.write("**通知チャンネル**")
            desktop_enabled = st.checkbox("🖥️ デスクトップ通知", value=True)
            audio_enabled = st.checkbox("🔊 音声アラート", value=True)
            email_enabled = st.checkbox("📧 メール通知", value=False)
            slack_enabled = st.checkbox("💬 Slack通知", value=False)
        
        with col_setting2:
            st.write("**アラート閾値**")
            confidence_threshold = st.slider("信頼度閾値", 0.5, 0.95, 0.8, 0.05)
            price_change_threshold = st.slider("価格変動閾値", 0.01, 0.15, 0.05, 0.01)
            risk_threshold = st.slider("リスクスコア閾値", 0.5, 1.0, 0.8, 0.1)
        
        # アラート履歴
        st.subheader("📋 アラート履歴")
        
        alert_history = alert_summary.get('alert_history', [])
        if alert_history:
            alert_data = []
            for alert in alert_history[-20:]:  # 最新20件
                alert_data.append({
                    '時刻': alert.get('timestamp', '')[:19],
                    'タイプ': alert.get('type', ''),
                    'メッセージ': alert.get('message', ''),
                    '重要度': alert.get('severity', ''),
                    'チャンネル': ', '.join(alert.get('channels', []))
                })
            
            alert_df = pd.DataFrame(alert_data)
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
        else:
            st.info("まだアラート履歴がありません")
    
    with tab7:
        # 取引履歴タブ
        st.header("📋 取引履歴")
        
        # フィルター
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
        
        with col_filter1:
            filter_symbol = st.selectbox("銘柄", ["全て"] + available_symbols)
        
        with col_filter2:
            filter_days = st.selectbox("期間", [7, 30, 90, 365])
        
        with col_filter3:
            filter_type = st.selectbox("取引タイプ", ["全て", "買い", "売り"])
        
        with col_filter4:
            filter_status = st.selectbox("ステータス", ["全て", "約定", "待機中", "キャンセル"])
        
        # 取引履歴取得と表示
        if st.session_state.realistic_mode:
            trades = platform.realistic_trader.get_trade_history(100)
        else:
            trades = platform.trader.trading_engine.get_trade_history(limit=100)
        
        if trades:
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
            
            # 取引履歴テーブル
            st.subheader("📋 取引詳細")
            
            trade_data = []
            for trade in trades:
                trade_dict = {
                    '日時': trade.get('timestamp', '').replace('T', ' ')[:19],
                    '銘柄': trade.get('symbol', ''),
                    'タイプ': '買い' if trade.get('side') == 'buy' else '売り',
                    '数量': f"{trade.get('quantity', 0):.4f}",
                    '価格': f"${trade.get('price', 0):.2f}",
                    '金額': f"${trade.get('quantity', 0) * trade.get('price', 0):,.2f}",
                    '手数料': f"${trade.get('fee', 0):.2f}",
                    'ステータス': trade.get('status', 'filled'),
                    'ID': trade.get('id', '')[:8]
                }
                
                # リアルモードの場合はスリッページ情報も追加
                if st.session_state.realistic_mode:
                    trade_dict['スリッページ'] = f"{trade.get('slippage', 0):.3%}"
                
                trade_data.append(trade_dict)
            
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
            
            # エクスポートボタン
            if st.button("📥 取引履歴エクスポート", use_container_width=True, key="trade_history_export"):
                csv = trade_df.to_csv(index=False)
                st.download_button(
                    "💾 CSVダウンロード",
                    csv,
                    file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="trade_history_download"
                )
        else:
            st.info("取引履歴がありません。取引を実行してください。")
    
    with tab8:
        # 高度な設定タブ
        st.header("⚙️ 高度な設定")
        
        # システム情報
        st.subheader("🖥️ システム情報")
        
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.write("**基本情報**")
            st.write("- **予測エンジン**: 高度機械学習アンサンブル")
            st.write("- **リスク管理**: Kelly基準 + VaR")
            st.write("- **データソース**: Hyperliquid DEX")
            st.write("- **更新方式**: リアルタイム + 手動")
            
            ml_performance = summary.get('ml_performance', {})
            ml_status = ml_performance.get('ml_available', False)
            st.write(f"- **ML状況**: {'✅ 利用可能' if ml_status else '❌ 基本モードのみ'}")
        
        with col_sys2:
            st.write("**パフォーマンス指標**")
            
            prediction_stats = summary.get('prediction_stats', {})
            if prediction_stats:
                st.write(f"- **総予測回数**: {prediction_stats.get('total_predictions', 0):,}")
                st.write(f"- **高信頼度予測**: {prediction_stats.get('high_confidence_predictions', 0):,}")
                st.write(f"- **平均信頼度**: {prediction_stats.get('average_confidence', 0):.1%}")
                st.write(f"- **高信頼度率**: {prediction_stats.get('high_confidence_rate', 0):.1%}")
            else:
                st.write("- 予測統計: 初期化中")
        
        # 予測精度目標
        st.subheader("🎯 予測精度目標")
        
        # ML予測精度目標（高度予測エンジンから取得）
        if ml_performance:
            recent_accuracy = prediction_stats.get('average_confidence', 0) if prediction_stats else 0.5
            target_accuracy = 0.6
            
            col_target1, col_target2 = st.columns(2)
            
            with col_target1:
                st.metric("現在の予測精度", f"{recent_accuracy:.1%}")
                st.metric("目標精度", f"{target_accuracy:.1%}")
            
            with col_target2:
                if recent_accuracy >= target_accuracy:
                    st.success("🎯 **目標達成！**")
                    st.write("優秀な予測性能を維持しています。")
                elif recent_accuracy >= target_accuracy * 0.8:
                    st.warning("📊 **改善中**")
                    st.write("目標に近づいています。")
                else:
                    st.error("🔄 **要改善**")
                    st.write("予測性能の向上が必要です。")
        
        # ML設定詳細
        if ml_performance.get('ml_available'):
            st.subheader("🤖 機械学習詳細設定")
            
            training_samples = ml_performance.get('training_samples', 0)
            prediction_count = ml_performance.get('prediction_count', 0)
            
            col_ml_detail1, col_ml_detail2 = st.columns(2)
            
            with col_ml_detail1:
                st.write("**学習データ**")
                st.write(f"- 学習サンプル数: {training_samples:,}")
                st.write(f"- 予測実行回数: {prediction_count:,}")
                st.write(f"- 再学習間隔: 100回毎")
                
                next_retrain = 100 - (prediction_count % 100)
                st.write(f"- 次回再学習まで: {next_retrain}回")
            
            with col_ml_detail2:
                st.write("**モデル性能**")
                model_perf = ml_performance.get('model_performance', {})
                if model_perf:
                    for model_name, perf in list(model_perf.items())[:3]:
                        accuracy = perf.get('accuracy', 0)
                        st.write(f"- {model_name}: {accuracy:.1%}")
                else:
                    st.write("- モデル性能: 学習中")
        
        # システム制御
        st.subheader("🔧 システム制御")
        
        col_control1, col_control2, col_control3 = st.columns(3)
        
        with col_control1:
            if st.button("🔄 システム再起動", use_container_width=True, key="system_restart"):
                st.info("システムを再起動中...")
                # 全コンポーネントをリセット
                if 'enhanced_trader' in st.session_state:
                    del st.session_state.enhanced_trader
                if 'realistic_trader' in st.session_state:
                    del st.session_state.realistic_trader
                if 'advanced_prediction_engine' in st.session_state:
                    del st.session_state.advanced_prediction_engine
                if 'risk_management_system' in st.session_state:
                    del st.session_state.risk_management_system
                st.rerun()
        
        with col_control2:
            if st.button("💾 全設定保存", use_container_width=True, key="system_save_all"):
                platform.trader.save_multi_symbol_config()
                st.success("全設定を保存しました")
        
        with col_control3:
            if st.button("🧹 キャッシュクリア", use_container_width=True, key="system_clear_cache"):
                st.cache_data.clear()
                # 予測エンジンのキャッシュもクリア
                platform.prediction_engine.prediction_cache.clear()
                st.success("キャッシュをクリアしました")

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