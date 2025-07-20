#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
パフォーマンス分析システム
取引結果の詳細分析とレポート生成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.io import to_image
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """時間枠"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL = "all"

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_return: float
    total_return_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_hold_time: float
    best_trade: Dict
    worst_trade: Dict
    consecutive_wins: int
    consecutive_losses: int

@dataclass
class SymbolPerformance:
    """銘柄別パフォーマンス"""
    symbol: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    total_volume: float
    avg_hold_time: float

class PerformanceAnalyzer:
    """パフォーマンス分析器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 図表設定
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # データ保存
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.predictions_history: List[Dict] = []
        
        logger.info("パフォーマンス分析システム初期化完了")
    
    def add_trade(self, trade: Dict):
        """取引記録追加"""
        self.trade_history.append(trade)
        logger.debug(f"取引記録追加: {trade.get('symbol', 'Unknown')}")
    
    def add_equity_point(self, timestamp: datetime, equity: float):
        """資産推移記録"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def add_prediction(self, prediction: Dict):
        """予測記録追加"""
        self.predictions_history.append(prediction)
    
    def calculate_metrics(self, trades: List[Dict], initial_balance: float = 10000) -> PerformanceMetrics:
        """パフォーマンス指標計算"""
        if not trades:
            return self._get_empty_metrics()
        
        # PnL計算
        pnls = [t.get('pnl', 0) for t in trades]
        returns = [p / initial_balance for p in pnls]
        
        # 勝敗分析
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_return = sum(pnls)
        total_return_pct = total_return / initial_balance * 100
        
        # 勝率
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # プロフィットファクター
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # シャープレシオ
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # ソルティノレシオ
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # 最大ドローダウン
        max_dd, max_dd_duration = self._calculate_max_drawdown(pnls, initial_balance)
        
        # 平均損益
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # 最大損益
        largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # 平均保有時間
        hold_times = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                hold_times.append(duration)
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        # 最良・最悪取引
        best_trade = max(trades, key=lambda x: x.get('pnl', 0)) if trades else {}
        worst_trade = min(trades, key=lambda x: x.get('pnl', 0)) if trades else {}
        
        # 連続勝敗
        consecutive_wins, consecutive_losses = self._calculate_streaks(trades)
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_hold_time=avg_hold_time,
            best_trade=best_trade,
            worst_trade=worst_trade,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )
    
    def analyze_by_symbol(self, trades: List[Dict]) -> List[SymbolPerformance]:
        """銘柄別分析"""
        symbol_groups = {}
        
        # 銘柄別グループ化
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(trade)
        
        # 銘柄別パフォーマンス計算
        performances = []
        for symbol, symbol_trades in symbol_groups.items():
            winning = len([t for t in symbol_trades if t.get('pnl', 0) > 0])
            total_pnl = sum(t.get('pnl', 0) for t in symbol_trades)
            volumes = [t.get('quantity', 0) * t.get('price', 0) for t in symbol_trades]
            
            # 保有時間計算
            hold_times = []
            for trade in symbol_trades:
                if 'entry_time' in trade and 'exit_time' in trade:
                    duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                    hold_times.append(duration)
            
            perf = SymbolPerformance(
                symbol=symbol,
                total_trades=len(symbol_trades),
                win_rate=winning / len(symbol_trades) if symbol_trades else 0,
                total_pnl=total_pnl,
                avg_pnl=total_pnl / len(symbol_trades) if symbol_trades else 0,
                best_trade=max(t.get('pnl', 0) for t in symbol_trades) if symbol_trades else 0,
                worst_trade=min(t.get('pnl', 0) for t in symbol_trades) if symbol_trades else 0,
                total_volume=sum(volumes),
                avg_hold_time=np.mean(hold_times) if hold_times else 0
            )
            performances.append(perf)
        
        return sorted(performances, key=lambda x: x.total_pnl, reverse=True)
    
    def analyze_by_timeframe(self, trades: List[Dict], timeframe: TimeFrame) -> Dict[str, PerformanceMetrics]:
        """時間枠別分析"""
        grouped_trades = self._group_trades_by_timeframe(trades, timeframe)
        
        results = {}
        for period, period_trades in grouped_trades.items():
            if period_trades:
                metrics = self.calculate_metrics(period_trades)
                results[period] = metrics
        
        return results
    
    def analyze_predictions(self) -> Dict:
        """予測精度分析"""
        if not self.predictions_history:
            return {}
        
        # 銘柄別精度
        symbol_accuracy = {}
        for pred in self.predictions_history:
            symbol = pred.get('symbol', 'Unknown')
            if symbol not in symbol_accuracy:
                symbol_accuracy[symbol] = {'correct': 0, 'total': 0}
            
            symbol_accuracy[symbol]['total'] += 1
            if pred.get('was_correct', False):
                symbol_accuracy[symbol]['correct'] += 1
        
        # 信頼度別精度
        confidence_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        confidence_accuracy = {}
        
        for low, high in confidence_bins:
            bin_preds = [p for p in self.predictions_history 
                        if low <= p.get('confidence', 0) < high]
            if bin_preds:
                correct = sum(1 for p in bin_preds if p.get('was_correct', False))
                confidence_accuracy[f"{low:.1f}-{high:.1f}"] = {
                    'accuracy': correct / len(bin_preds),
                    'count': len(bin_preds)
                }
        
        # シグナル別精度
        signal_accuracy = {}
        for signal in ['BUY', 'SELL', 'HOLD']:
            signal_preds = [p for p in self.predictions_history 
                          if p.get('signal') == signal]
            if signal_preds:
                correct = sum(1 for p in signal_preds if p.get('was_correct', False))
                signal_accuracy[signal] = {
                    'accuracy': correct / len(signal_preds),
                    'count': len(signal_preds)
                }
        
        return {
            'total_predictions': len(self.predictions_history),
            'overall_accuracy': sum(1 for p in self.predictions_history if p.get('was_correct', False)) / len(self.predictions_history),
            'symbol_accuracy': symbol_accuracy,
            'confidence_accuracy': confidence_accuracy,
            'signal_accuracy': signal_accuracy
        }
    
    def create_performance_charts(self) -> Dict[str, go.Figure]:
        """パフォーマンスチャート作成"""
        charts = {}
        
        # 1. 資産推移チャート
        if self.equity_curve:
            fig = go.Figure()
            
            timestamps = [e['timestamp'] for e in self.equity_curve]
            equities = [e['equity'] for e in self.equity_curve]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=equities,
                mode='lines',
                name='資産推移',
                line=dict(color='blue', width=2)
            ))
            
            # ドローダウン領域
            peak = equities[0]
            drawdowns = []
            for equity in equities:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                drawdowns.append(drawdown)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[-dd for dd in drawdowns],
                mode='lines',
                name='ドローダウン',
                fill='tozeroy',
                line=dict(color='red', width=1),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='資産推移とドローダウン',
                xaxis_title='日時',
                yaxis_title='資産 ($)',
                yaxis2=dict(
                    title='ドローダウン (%)',
                    overlaying='y',
                    side='right'
                ),
                template='plotly_dark',
                height=500
            )
            
            charts['equity_curve'] = fig
        
        # 2. 月次リターンヒートマップ
        if self.trade_history:
            monthly_returns = self._calculate_monthly_returns()
            
            if monthly_returns:
                years = sorted(set(date.year for date in monthly_returns.keys()))
                months = list(range(1, 13))
                
                z_data = []
                for year in years:
                    year_data = []
                    for month in months:
                        key = datetime(year, month, 1).date()
                        year_data.append(monthly_returns.get(key, 0))
                    z_data.append(year_data)
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=years,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f"{val:.1f}%" for val in row] for row in z_data],
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title='月次リターンヒートマップ',
                    xaxis_title='月',
                    yaxis_title='年',
                    template='plotly_dark',
                    height=400
                )
                
                charts['monthly_returns'] = fig
        
        # 3. 銘柄別パフォーマンス
        symbol_perfs = self.analyze_by_symbol(self.trade_history)
        if symbol_perfs:
            fig = go.Figure()
            
            symbols = [p.symbol for p in symbol_perfs]
            pnls = [p.total_pnl for p in symbol_perfs]
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=pnls,
                marker_color=colors,
                text=[f"${pnl:.2f}" for pnl in pnls],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='銘柄別損益',
                xaxis_title='銘柄',
                yaxis_title='総損益 ($)',
                template='plotly_dark',
                height=400
            )
            
            charts['symbol_performance'] = fig
        
        # 4. 損益分布
        if self.trade_history:
            pnls = [t.get('pnl', 0) for t in self.trade_history]
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=pnls,
                nbinsx=30,
                name='損益分布',
                marker_color='blue'
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            fig.add_vline(x=np.mean(pnls), line_dash="dash", line_color="green", 
                         annotation_text=f"平均: ${np.mean(pnls):.2f}")
            
            fig.update_layout(
                title='損益分布',
                xaxis_title='損益 ($)',
                yaxis_title='頻度',
                template='plotly_dark',
                height=400
            )
            
            charts['pnl_distribution'] = fig
        
        return charts
    
    def generate_html_report(self, filename: str = None) -> str:
        """HTMLレポート生成"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = self.output_dir / filename
        
        # メトリクス計算
        metrics = self.calculate_metrics(self.trade_history)
        symbol_perfs = self.analyze_by_symbol(self.trade_history)
        prediction_analysis = self.analyze_predictions()
        charts = self.create_performance_charts()
        
        # HTML生成
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Performance Report</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }}
                .header {{
                    text-align: center;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #2d2d2d;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #888;
                    margin-top: 5px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #2d2d2d;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .table th, .table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #444;
                }}
                .table th {{
                    background-color: #3d3d3d;
                    font-weight: bold;
                }}
                .positive {{
                    color: #4CAF50;
                }}
                .negative {{
                    color: #f44336;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Trading Performance Report</h1>
                <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 主要パフォーマンス指標</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value {'positive' if metrics.total_return >= 0 else 'negative'}">
                            ${metrics.total_return:,.2f}
                        </div>
                        <div class="metric-label">総損益</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {'positive' if metrics.total_return_pct >= 0 else 'negative'}">
                            {metrics.total_return_pct:.2f}%
                        </div>
                        <div class="metric-label">総リターン率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.win_rate:.1f}%</div>
                        <div class="metric-label">勝率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.profit_factor:.2f}</div>
                        <div class="metric-label">プロフィットファクター</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                        <div class="metric-label">シャープレシオ</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value negative">{metrics.max_drawdown:.1f}%</div>
                        <div class="metric-label">最大ドローダウン</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.total_trades}</div>
                        <div class="metric-label">総取引数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.avg_hold_time:.1f}h</div>
                        <div class="metric-label">平均保有時間</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 資産推移</h2>
                <div class="chart-container" id="equity_curve"></div>
            </div>
            
            <div class="section">
                <h2>🔥 月次リターン</h2>
                <div class="chart-container" id="monthly_returns"></div>
            </div>
            
            <div class="section">
                <h2>💰 銘柄別パフォーマンス</h2>
                <div class="chart-container" id="symbol_performance"></div>
                
                <table class="table" style="margin-top: 20px;">
                    <thead>
                        <tr>
                            <th>銘柄</th>
                            <th>取引数</th>
                            <th>勝率</th>
                            <th>総損益</th>
                            <th>平均損益</th>
                            <th>最大利益</th>
                            <th>最大損失</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # 銘柄別テーブル
        for perf in symbol_perfs[:10]:  # Top 10
            html_content += f"""
                        <tr>
                            <td>{perf.symbol}</td>
                            <td>{perf.total_trades}</td>
                            <td>{perf.win_rate:.1f}%</td>
                            <td class="{'positive' if perf.total_pnl >= 0 else 'negative'}">${perf.total_pnl:.2f}</td>
                            <td class="{'positive' if perf.avg_pnl >= 0 else 'negative'}">${perf.avg_pnl:.2f}</td>
                            <td class="positive">${perf.best_trade:.2f}</td>
                            <td class="negative">${perf.worst_trade:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📊 損益分布</h2>
                <div class="chart-container" id="pnl_distribution"></div>
            </div>
        """
        
        # 予測精度セクション
        if prediction_analysis:
            html_content += f"""
            <div class="section">
                <h2>🎯 予測精度分析</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{prediction_analysis.get('overall_accuracy', 0):.1%}</div>
                        <div class="metric-label">全体精度</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{prediction_analysis.get('total_predictions', 0)}</div>
                        <div class="metric-label">総予測数</div>
                    </div>
                </div>
            </div>
            """
        
        # チャートスクリプト
        html_content += """
            <script>
        """
        
        for chart_id, chart in charts.items():
            html_content += f"""
                Plotly.newPlot('{chart_id}', {chart.to_json()});
            """
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # ファイル保存
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTMLレポート生成完了: {filepath}")
        return str(filepath)
    
    def generate_pdf_report(self, filename: str = None) -> str:
        """PDFレポート生成"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        filepath = self.output_dir / filename
        
        # PDFドキュメント作成
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # タイトル
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a73e8'),
            spaceAfter=30
        )
        story.append(Paragraph("AI Trading Performance Report", title_style))
        story.append(Spacer(1, 12))
        
        # 生成日時
        story.append(Paragraph(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # メトリクス計算
        metrics = self.calculate_metrics(self.trade_history)
        
        # 主要指標テーブル
        data = [
            ['指標', '値'],
            ['総損益', f'${metrics.total_return:,.2f}'],
            ['総リターン率', f'{metrics.total_return_pct:.2f}%'],
            ['勝率', f'{metrics.win_rate:.1f}%'],
            ['総取引数', f'{metrics.total_trades}'],
            ['勝ち取引', f'{metrics.winning_trades}'],
            ['負け取引', f'{metrics.losing_trades}'],
            ['プロフィットファクター', f'{metrics.profit_factor:.2f}'],
            ['シャープレシオ', f'{metrics.sharpe_ratio:.2f}'],
            ['最大ドローダウン', f'{metrics.max_drawdown:.1f}%'],
            ['平均利益', f'${metrics.avg_win:.2f}'],
            ['平均損失', f'${metrics.avg_loss:.2f}'],
            ['最大利益', f'${metrics.largest_win:.2f}'],
            ['最大損失', f'${metrics.largest_loss:.2f}'],
            ['平均保有時間', f'{metrics.avg_hold_time:.1f}時間'],
            ['連続勝利', f'{metrics.consecutive_wins}'],
            ['連続敗北', f'{metrics.consecutive_losses}']
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("主要パフォーマンス指標", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(table)
        story.append(PageBreak())
        
        # 銘柄別パフォーマンス
        symbol_perfs = self.analyze_by_symbol(self.trade_history)
        if symbol_perfs:
            story.append(Paragraph("銘柄別パフォーマンス", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            symbol_data = [['銘柄', '取引数', '勝率', '総損益', '平均損益']]
            for perf in symbol_perfs[:10]:
                symbol_data.append([
                    perf.symbol,
                    str(perf.total_trades),
                    f'{perf.win_rate:.1f}%',
                    f'${perf.total_pnl:.2f}',
                    f'${perf.avg_pnl:.2f}'
                ])
            
            symbol_table = Table(symbol_data)
            symbol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(symbol_table)
        
        # PDF生成
        doc.build(story)
        
        logger.info(f"PDFレポート生成完了: {filepath}")
        return str(filepath)
    
    def export_trade_history(self, filename: str = None, format: str = 'csv') -> str:
        """取引履歴エクスポート"""
        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        filepath = self.output_dir / filename
        
        df = pd.DataFrame(self.trade_history)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        
        logger.info(f"取引履歴エクスポート完了: {filepath}")
        return str(filepath)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """シャープレシオ計算"""
        if not returns:
            return 0
        
        excess_returns = [r - risk_free_rate / 252 for r in returns]  # 日次換算
        
        if len(excess_returns) < 2:
            return 0
        
        avg_excess_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0
        
        return avg_excess_return / std_return * np.sqrt(252)  # 年率化
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """ソルティノレシオ計算"""
        if not returns:
            return 0
        
        excess_returns = [r - risk_free_rate / 252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns or len(returns) < 2:
            return 0
        
        avg_excess_return = np.mean(excess_returns)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return avg_excess_return / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, pnls: List[float], initial_balance: float) -> Tuple[float, int]:
        """最大ドローダウン計算"""
        if not pnls:
            return 0, 0
        
        equity = initial_balance
        peak = equity
        max_dd = 0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for pnl in pnls:
            equity += pnl
            
            if equity > peak:
                peak = equity
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                drawdown = (peak - equity) / peak * 100
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_duration = current_dd_duration
        
        return max_dd, max_dd_duration
    
    def _calculate_streaks(self, trades: List[Dict]) -> Tuple[int, int]:
        """連続勝敗計算"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """空のメトリクス"""
        return PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            win_rate=0,
            profit_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_hold_time=0,
            best_trade={},
            worst_trade={},
            consecutive_wins=0,
            consecutive_losses=0
        )
    
    def _group_trades_by_timeframe(self, trades: List[Dict], timeframe: TimeFrame) -> Dict[str, List[Dict]]:
        """時間枠別グループ化"""
        grouped = {}
        
        for trade in trades:
            timestamp = trade.get('timestamp', trade.get('exit_time'))
            if not timestamp:
                continue
            
            if timeframe == TimeFrame.DAILY:
                key = timestamp.date().isoformat()
            elif timeframe == TimeFrame.WEEKLY:
                key = f"{timestamp.year}-W{timestamp.isocalendar()[1]:02d}"
            elif timeframe == TimeFrame.MONTHLY:
                key = f"{timestamp.year}-{timestamp.month:02d}"
            elif timeframe == TimeFrame.YEARLY:
                key = str(timestamp.year)
            else:  # ALL
                key = "all"
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(trade)
        
        return grouped
    
    def _calculate_monthly_returns(self) -> Dict[datetime, float]:
        """月次リターン計算"""
        if not self.trade_history:
            return {}
        
        monthly_pnls = {}
        
        for trade in self.trade_history:
            timestamp = trade.get('timestamp', trade.get('exit_time'))
            if timestamp:
                month_key = datetime(timestamp.year, timestamp.month, 1).date()
                if month_key not in monthly_pnls:
                    monthly_pnls[month_key] = 0
                monthly_pnls[month_key] += trade.get('pnl', 0)
        
        # パーセンテージに変換（仮定の月初資産10000）
        monthly_returns = {
            month: pnl / 10000 * 100 for month, pnl in monthly_pnls.items()
        }
        
        return monthly_returns

# テスト関数
def test_performance_analyzer():
    """パフォーマンス分析システムのテスト"""
    print("=== パフォーマンス分析システムテスト ===")
    
    analyzer = PerformanceAnalyzer()
    
    # サンプル取引データ生成
    base_time = datetime.now() - timedelta(days=30)
    sample_trades = []
    
    for i in range(50):
        entry_time = base_time + timedelta(hours=i*12)
        exit_time = entry_time + timedelta(hours=np.random.randint(1, 24))
        pnl = np.random.normal(50, 200)
        
        trade = {
            'symbol': np.random.choice(['BTC', 'ETH', 'SOL', 'AVAX']),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'timestamp': exit_time,
            'side': np.random.choice(['buy', 'sell']),
            'quantity': np.random.uniform(0.01, 1),
            'entry_price': np.random.uniform(20000, 70000),
            'exit_price': np.random.uniform(20000, 70000),
            'pnl': pnl,
            'fee': np.random.uniform(1, 10)
        }
        
        sample_trades.append(trade)
        analyzer.add_trade(trade)
        
        # 資産推移
        equity = 10000 + sum(t['pnl'] for t in sample_trades[:i+1])
        analyzer.add_equity_point(exit_time, equity)
    
    # メトリクス計算
    metrics = analyzer.calculate_metrics(sample_trades)
    print(f"\n総損益: ${metrics.total_return:.2f}")
    print(f"勝率: {metrics.win_rate:.1f}%")
    print(f"プロフィットファクター: {metrics.profit_factor:.2f}")
    print(f"シャープレシオ: {metrics.sharpe_ratio:.2f}")
    print(f"最大ドローダウン: {metrics.max_drawdown:.1f}%")
    
    # 銘柄別分析
    symbol_perfs = analyzer.analyze_by_symbol(sample_trades)
    print("\n銘柄別パフォーマンス:")
    for perf in symbol_perfs:
        print(f"  {perf.symbol}: 取引{perf.total_trades}回, PnL ${perf.total_pnl:.2f}")
    
    # レポート生成
    print("\nレポート生成中...")
    html_report = analyzer.generate_html_report()
    print(f"HTMLレポート: {html_report}")
    
    pdf_report = analyzer.generate_pdf_report()
    print(f"PDFレポート: {pdf_report}")
    
    # 取引履歴エクスポート
    csv_export = analyzer.export_trade_history(format='csv')
    print(f"取引履歴CSV: {csv_export}")

if __name__ == "__main__":
    test_performance_analyzer()