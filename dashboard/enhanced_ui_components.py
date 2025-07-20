#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拡張UIコンポーネント
ユーザビリティ向上のための追加コンポーネント
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

from core.error_handler import safe_execute, create_error_boundary

@create_error_boundary("ローディングアニメーション")
def show_loading_animation(message: str = "データを読み込み中..."):
    """ローディングアニメーション表示"""
    placeholder = st.empty()
    
    for i in range(3):
        dots = "." * (i + 1)
        placeholder.info(f"⏳ {message}{dots}")
        time.sleep(0.5)
    
    placeholder.empty()

@create_error_boundary("成功通知")
def show_success_notification(message: str, duration: float = 3.0):
    """成功通知を表示"""
    success_placeholder = st.empty()
    success_placeholder.success(f"✅ {message}")
    
    # 一定時間後に自動で消す
    if duration > 0:
        time.sleep(duration)
        success_placeholder.empty()

@create_error_boundary("警告通知")
def show_warning_notification(message: str, duration: float = 5.0):
    """警告通知を表示"""
    warning_placeholder = st.empty()
    warning_placeholder.warning(f"⚠️ {message}")
    
    if duration > 0:
        time.sleep(duration)
        warning_placeholder.empty()

@create_error_boundary("ヘルプツールチップ")
def create_help_tooltip(content: str, help_text: str):
    """ヘルプ付きコンテンツ"""
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        st.write(content)
    
    with col2:
        if st.button("❓", key=f"help_{hash(content)}", help=help_text):
            st.info(help_text)

@create_error_boundary("プログレスバー")
class ProgressTracker:
    """プログレス表示クラス"""
    
    def __init__(self, total_steps: int, title: str = "進行状況"):
        self.total_steps = total_steps
        self.current_step = 0
        self.title = title
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step_name: str = ""):
        """プログレスを更新"""
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        
        self.progress_bar.progress(progress)
        
        if step_name:
            self.status_text.text(f"{self.title}: {step_name} ({self.current_step}/{self.total_steps})")
        else:
            self.status_text.text(f"{self.title}: {self.current_step}/{self.total_steps}")
    
    def complete(self, message: str = "完了"):
        """プログレス完了"""
        self.progress_bar.progress(1.0)
        self.status_text.success(f"✅ {message}")
        
        # 2秒後にクリア
        time.sleep(2)
        self.progress_bar.empty()
        self.status_text.empty()

@create_error_boundary("拡張メトリクス")
def create_enhanced_metric_card(
    title: str,
    value: Any,
    delta: Any = None,
    delta_color: str = "normal",
    help_text: str = None,
    icon: str = "📊",
    background_color: str = "#1e1e1e"
):
    """拡張メトリクスカード"""
    
    # カスタムスタイル
    card_style = f"""
    <div style="
        background: linear-gradient(145deg, {background_color}, #2d2d2d);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00D4AA;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <h4 style="margin: 0; color: #ffffff; font-size: 0.9rem;">{title}</h4>
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #00D4AA;">
            {value}
        </div>
        {f'<div style="font-size: 0.8rem; color: {"#00D4AA" if delta_color == "normal" else "#FF6B6B"};">{delta}</div>' if delta else ''}
        {f'<div style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">{help_text}</div>' if help_text else ''}
    </div>
    """
    
    st.markdown(card_style, unsafe_allow_html=True)

@create_error_boundary("インタラクティブチャート")
def create_interactive_performance_chart(data: Dict[str, List], title: str = "パフォーマンス") -> go.Figure:
    """インタラクティブなパフォーマンスチャート"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("損益推移", "勝率", "リスク指標", "取引量"),
        specs=[[{"secondary_y": True}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. 損益推移（時系列）
    if 'pnl_history' in data:
        fig.add_trace(
            go.Scatter(
                x=data.get('dates', []),
                y=data['pnl_history'],
                mode='lines+markers',
                name='累積損益',
                line=dict(color='#00D4AA', width=2)
            ),
            row=1, col=1
        )
    
    # 2. 勝率ゲージ
    if 'win_rate' in data:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data['win_rate'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "勝率 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00D4AA"},
                    'steps': [
                        {'range': [0, 25], 'color': "#FF6B6B"},
                        {'range': [25, 50], 'color': "#FFD700"},
                        {'range': [50, 75], 'color': "#90EE90"},
                        {'range': [75, 100], 'color': "#00D4AA"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=2
        )
    
    # 3. リスク指標バー
    if 'risk_metrics' in data:
        risk_names = list(data['risk_metrics'].keys())
        risk_values = list(data['risk_metrics'].values())
        
        fig.add_trace(
            go.Bar(
                x=risk_names,
                y=risk_values,
                name='リスク指標',
                marker_color=['#FF6B6B' if v > 0.7 else '#FFD700' if v > 0.4 else '#00D4AA' for v in risk_values]
            ),
            row=2, col=1
        )
    
    # 4. 取引量散布図
    if 'trade_volumes' in data and 'trade_profits' in data:
        fig.add_trace(
            go.Scatter(
                x=data['trade_volumes'],
                y=data['trade_profits'],
                mode='markers',
                name='取引',
                marker=dict(
                    size=8,
                    color=data['trade_profits'],
                    colorscale='RdYlGn',
                    showscale=True
                )
            ),
            row=2, col=2
        )
    
    # レイアウト更新
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    return fig

@create_error_boundary("アラート管理")
class AlertManager:
    """アラート管理システム"""
    
    def __init__(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def add_alert(self, message: str, alert_type: str = "info", duration: int = 5):
        """アラートを追加"""
        alert = {
            'id': time.time(),
            'message': message,
            'type': alert_type,
            'timestamp': datetime.now(),
            'duration': duration
        }
        st.session_state.alerts.append(alert)
    
    def show_alerts(self):
        """アラートを表示"""
        current_time = datetime.now()
        active_alerts = []
        
        for alert in st.session_state.alerts:
            time_diff = (current_time - alert['timestamp']).total_seconds()
            
            if time_diff < alert['duration']:
                active_alerts.append(alert)
                
                # アラート表示
                if alert['type'] == 'success':
                    st.success(f"✅ {alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"⚠️ {alert['message']}")
                elif alert['type'] == 'error':
                    st.error(f"❌ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        
        # 期限切れアラートを削除
        st.session_state.alerts = active_alerts

@create_error_boundary("データテーブル")
def create_enhanced_data_table(
    data: pd.DataFrame,
    title: str = "データテーブル",
    searchable: bool = True,
    sortable: bool = True,
    page_size: int = 10
):
    """拡張データテーブル"""
    
    st.subheader(title)
    
    if data.empty:
        st.info("📋 表示するデータがありません")
        return
    
    # 検索機能
    if searchable:
        search_term = st.text_input("🔍 検索", placeholder="キーワードを入力...")
        if search_term:
            # 文字列列での検索
            string_cols = data.select_dtypes(include=['object']).columns
            mask = data[string_cols].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            data = data[mask]
    
    # ソート機能
    if sortable and not data.empty:
        sort_col = st.selectbox("📊 ソート列", ["なし"] + list(data.columns))
        if sort_col != "なし":
            sort_order = st.radio("順序", ["昇順", "降順"], horizontal=True)
            ascending = sort_order == "昇順"
            data = data.sort_values(sort_col, ascending=ascending)
    
    # ページネーション
    if len(data) > page_size:
        total_pages = (len(data) - 1) // page_size + 1
        page = st.selectbox(f"📄 ページ (全{total_pages}ページ)", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        data = data.iloc[start_idx:end_idx]
    
    # テーブル表示
    st.dataframe(
        data,
        use_container_width=True,
        height=min(400, len(data) * 35 + 50)
    )
    
    # 統計情報
    with st.expander("📊 統計情報"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("行数", len(data))
        
        with col2:
            st.metric("列数", len(data.columns))
        
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.metric("数値列", len(numeric_cols))

@create_error_boundary("ツアーガイド")
class TourGuide:
    """アプリケーションツアーガイド"""
    
    def __init__(self):
        if 'tour_completed' not in st.session_state:
            st.session_state.tour_completed = False
        if 'tour_step' not in st.session_state:
            st.session_state.tour_step = 0
    
    def show_welcome_tour(self):
        """ウェルカムツアーを表示"""
        if not st.session_state.tour_completed:
            
            if st.session_state.tour_step == 0:
                st.info("""
                🎉 **統合AI取引プラットフォームへようこそ！**
                
                このプラットフォームでは以下の機能が利用できます：
                - 🤖 AI予測エンジン
                - 📊 リアルタイムチャート
                - 💰 ペーパートレーディング
                - 🛡️ リスク管理
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("ツアーを開始"):
                        st.session_state.tour_step = 1
                        st.rerun()
                
                with col2:
                    if st.button("スキップ"):
                        st.session_state.tour_completed = True
                        st.rerun()
            
            elif st.session_state.tour_step == 1:
                st.info("""
                📊 **メインダッシュボード**
                
                ここでは暗号通貨の価格チャートとAI予測を確認できます。
                左側のサイドバーで銘柄や時間枠を変更できます。
                """)
                
                if st.button("次へ"):
                    st.session_state.tour_step = 2
                    st.rerun()
            
            elif st.session_state.tour_step == 2:
                st.info("""
                🤖 **AI予測機能**
                
                当プラットフォームのAIは複数の機械学習モデルを使用して
                価格予測を行います。予測の信頼度も表示されます。
                """)
                
                if st.button("次へ"):
                    st.session_state.tour_step = 3
                    st.rerun()
            
            elif st.session_state.tour_step == 3:
                st.info("""
                💰 **取引機能**
                
                ペーパートレーディングで安全に取引を練習できます。
                リアルな市場条件を再現しています。
                """)
                
                if st.button("ツアー完了"):
                    st.session_state.tour_completed = True
                    st.success("ツアー完了！取引を始めましょう！")
                    st.rerun()

@create_error_boundary("キーボードショートカット")
def show_keyboard_shortcuts():
    """キーボードショートカットヘルプ"""
    with st.expander("⌨️ キーボードショートカット"):
        shortcuts = {
            "Ctrl + R": "ページ更新",
            "Ctrl + F": "検索",
            "Esc": "モーダル閉じる",
            "Space": "チャート一時停止/再開",
            "↑/↓": "銘柄選択",
            "←/→": "時間枠変更"
        }
        
        for key, description in shortcuts.items():
            st.write(f"**{key}**: {description}")

@create_error_boundary("パフォーマンス監視")
class PerformanceMonitor:
    """パフォーマンス監視"""
    
    def __init__(self):
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'load_times': [],
                'api_response_times': [],
                'error_counts': 0
            }
    
    def track_load_time(self, operation: str, duration: float):
        """ロード時間を記録"""
        st.session_state.performance_metrics['load_times'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # 最新100件のみ保持
        if len(st.session_state.performance_metrics['load_times']) > 100:
            st.session_state.performance_metrics['load_times'].pop(0)
    
    def show_performance_dashboard(self):
        """パフォーマンスダッシュボードを表示"""
        with st.expander("📈 パフォーマンス監視"):
            metrics = st.session_state.performance_metrics
            
            if metrics['load_times']:
                avg_load_time = np.mean([m['duration'] for m in metrics['load_times']])
                st.metric("平均ロード時間", f"{avg_load_time:.2f}秒")
                
                # ロード時間の推移グラフ
                load_df = pd.DataFrame(metrics['load_times'])
                if not load_df.empty:
                    fig = px.line(
                        load_df, 
                        x='timestamp', 
                        y='duration',
                        title='ロード時間推移',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # テスト実行
    st.set_page_config(page_title="UI Components Test", layout="wide")
    
    st.title("🎨 拡張UIコンポーネント テスト")
    
    # ツアーガイド
    tour = TourGuide()
    tour.show_welcome_tour()
    
    # メトリクスカード
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_enhanced_metric_card(
            "総利益", "$1,234.56", "+12.3%", "normal", 
            "今月の累積利益", "💰", "#1e4d1e"
        )
    
    with col2:
        create_enhanced_metric_card(
            "勝率", "68.5%", "+5.2%", "normal",
            "最近30取引の勝率", "🎯"
        )
    
    with col3:
        create_enhanced_metric_card(
            "リスク", "低", "-2.1%", "inverse",
            "現在のリスクレベル", "🛡️", "#4d1e1e"
        )
    
    # アラート管理
    alert_manager = AlertManager()
    alert_manager.show_alerts()
    
    if st.button("テストアラート"):
        alert_manager.add_alert("テストメッセージです", "success", 3)
        st.rerun()
    
    # キーボードショートカット
    show_keyboard_shortcuts()
    
    # パフォーマンス監視
    perf_monitor = PerformanceMonitor()
    perf_monitor.show_performance_dashboard()