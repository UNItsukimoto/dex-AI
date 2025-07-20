#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動テスト結果分析システム
テスト結果を詳細に分析し、視覚的なレポートを生成
"""

import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

class AutomatedTestAnalyzer:
    """自動テスト結果分析システム"""
    
    def __init__(self, db_path: str = "paper_test_results.db"):
        self.db_path = Path(db_path)
        self.results_dir = Path("test_analysis_reports")
        self.results_dir.mkdir(exist_ok=True)
        
        # データ読み込み
        self.trades_df = None
        self.performance_df = None
        self._load_data()
    
    def _load_data(self):
        """データベースからデータを読み込み"""
        if not self.db_path.exists():
            print("データベースが見つかりません")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # 取引データ
        self.trades_df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY timestamp", 
            conn
        )
        if not self.trades_df.empty:
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        
        # パフォーマンスデータ
        self.performance_df = pd.read_sql_query(
            "SELECT * FROM performance ORDER BY date", 
            conn
        )
        if not self.performance_df.empty:
            self.performance_df['date'] = pd.to_datetime(self.performance_df['date'])
        
        conn.close()
    
    def generate_comprehensive_report(self):
        """包括的な分析レポートを生成"""
        print("="*60)
        print("📊 自動テスト結果分析レポート生成")
        print("="*60)
        
        # 1. 基本統計
        self._analyze_basic_statistics()
        
        # 2. 戦略別詳細分析
        self._analyze_strategy_performance()
        
        # 3. 時系列分析
        self._analyze_time_series()
        
        # 4. リスク分析
        self._analyze_risk_metrics()
        
        # 5. 取引パターン分析
        self._analyze_trading_patterns()
        
        # 6. インタラクティブダッシュボード生成
        self._create_interactive_dashboard()
        
        print("\n✅ レポート生成完了")
        print(f"保存先: {self.results_dir}")
    
    def _analyze_basic_statistics(self):
        """基本統計分析"""
        print("\n📈 基本統計分析...")
        
        if self.trades_df.empty:
            print("取引データがありません")
            return
        
        # 戦略別統計
        strategy_stats = self.trades_df.groupby('strategy').agg({
            'id': 'count',
            'pnl': ['sum', 'mean', 'std'],
            'confidence': 'mean'
        }).round(2)
        
        # 結果をCSVで保存
        strategy_stats.to_csv(self.results_dir / 'strategy_statistics.csv')
        
        # サマリー表示
        print("\n戦略別取引統計:")
        print(strategy_stats)
    
    def _analyze_strategy_performance(self):
        """戦略別詳細パフォーマンス分析"""
        print("\n📊 戦略別パフォーマンス分析...")
        
        if self.performance_df.empty:
            return
        
        # 戦略別パフォーマンス推移グラフ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        strategies = self.performance_df['strategy'].unique()
        
        # 1. 累積損益
        ax = axes[0, 0]
        for strategy in strategies:
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            ax.plot(data['date'], data['total_pnl'].cumsum(), label=strategy.upper(), linewidth=2)
        ax.set_title('累積損益推移', fontsize=14)
        ax.set_xlabel('日付')
        ax.set_ylabel('累積損益 ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 勝率推移
        ax = axes[0, 1]
        for strategy in strategies:
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            ax.plot(data['date'], data['win_rate'], label=strategy.upper(), linewidth=2)
        ax.set_title('勝率推移', fontsize=14)
        ax.set_xlabel('日付')
        ax.set_ylabel('勝率 (%)')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. シャープレシオ
        ax = axes[1, 0]
        for strategy in strategies:
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            ax.plot(data['date'], data['sharpe_ratio'], label=strategy.upper(), linewidth=2)
        ax.set_title('シャープレシオ推移', fontsize=14)
        ax.set_xlabel('日付')
        ax.set_ylabel('シャープレシオ')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 最大ドローダウン
        ax = axes[1, 1]
        for strategy in strategies:
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            ax.plot(data['date'], data['max_drawdown'] * 100, label=strategy.upper(), linewidth=2)
        ax.set_title('最大ドローダウン推移', fontsize=14)
        ax.set_xlabel('日付')
        ax.set_ylabel('最大ドローダウン (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_time_series(self):
        """時系列分析"""
        print("\n⏰ 時系列分析...")
        
        if self.trades_df.empty:
            return
        
        # 時間帯別取引分析
        self.trades_df['hour'] = self.trades_df['timestamp'].dt.hour
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 時間帯別取引数
        hourly_trades = self.trades_df.groupby(['hour', 'strategy']).size().unstack()
        hourly_trades.plot(kind='bar', ax=ax1, stacked=True)
        ax1.set_title('時間帯別取引数', fontsize=14)
        ax1.set_xlabel('時間 (UTC)')
        ax1.set_ylabel('取引数')
        
        # 時間帯別平均損益
        hourly_pnl = self.trades_df.groupby('hour')['pnl'].mean()
        ax2.bar(hourly_pnl.index, hourly_pnl.values)
        ax2.set_title('時間帯別平均損益', fontsize=14)
        ax2.set_xlabel('時間 (UTC)')
        ax2.set_ylabel('平均損益 ($)')
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_risk_metrics(self):
        """リスク指標分析"""
        print("\n🛡️ リスク分析...")
        
        if self.performance_df.empty:
            return
        
        # リスク・リターンマトリックス
        fig = plt.figure(figsize=(10, 8))
        
        strategies = []
        returns = []
        risks = []
        sharpes = []
        
        for strategy in self.performance_df['strategy'].unique():
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            
            # 最終的なリターン
            final_pnl = data['total_pnl'].iloc[-1] if len(data) > 0 else 0
            ret = (final_pnl / 10000) * 100  # 初期資金10000と仮定
            
            # リスク（最大ドローダウン）
            risk = data['max_drawdown'].max() * 100
            
            # シャープレシオ
            sharpe = data['sharpe_ratio'].mean()
            
            strategies.append(strategy.upper())
            returns.append(ret)
            risks.append(risk)
            sharpes.append(sharpe)
        
        # バブルチャート（シャープレシオをサイズで表現）
        scatter = plt.scatter(risks, returns, s=np.array(sharpes)*200, alpha=0.6)
        
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (risks[i], returns[i]), ha='center', va='center')
        
        plt.xlabel('リスク (最大ドローダウン %)', fontsize=12)
        plt.ylabel('リターン (%)', fontsize=12)
        plt.title('リスク・リターン分析（バブルサイズ = シャープレシオ）', fontsize=14)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 効率的フロンティア風の曲線
        if len(risks) > 2:
            from scipy.interpolate import interp1d
            sorted_indices = np.argsort(risks)
            f = interp1d(np.array(risks)[sorted_indices], 
                        np.array(returns)[sorted_indices], 
                        kind='quadratic', fill_value='extrapolate')
            x_smooth = np.linspace(min(risks), max(risks), 100)
            y_smooth = f(x_smooth)
            plt.plot(x_smooth, y_smooth, 'g--', alpha=0.5, label='効率的フロンティア（推定）')
        
        plt.legend()
        plt.savefig(self.results_dir / 'risk_return_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_trading_patterns(self):
        """取引パターン分析"""
        print("\n🔍 取引パターン分析...")
        
        if self.trades_df.empty:
            return
        
        # 銘柄別分析
        symbol_analysis = self.trades_df.groupby(['symbol', 'strategy']).agg({
            'id': 'count',
            'pnl': 'sum'
        }).round(2)
        
        # ヒートマップ作成
        pivot_table = symbol_analysis['pnl'].unstack(fill_value=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn', center=0)
        plt.title('銘柄×戦略 損益ヒートマップ', fontsize=14)
        plt.xlabel('戦略')
        plt.ylabel('銘柄')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'symbol_strategy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_dashboard(self):
        """インタラクティブダッシュボード生成"""
        print("\n🎯 インタラクティブダッシュボード生成...")
        
        if self.performance_df.empty:
            return
        
        # Plotlyでインタラクティブな複合グラフ作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('累積損益', '勝率推移', 'ドローダウン', '取引分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        strategies = self.performance_df['strategy'].unique()
        colors = px.colors.qualitative.Set1
        
        # 1. 累積損益
        for i, strategy in enumerate(strategies):
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            fig.add_trace(
                go.Scatter(x=data['date'], y=data['total_pnl'].cumsum(),
                          mode='lines', name=strategy.upper(),
                          line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )
        
        # 2. 勝率推移
        for i, strategy in enumerate(strategies):
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            fig.add_trace(
                go.Scatter(x=data['date'], y=data['win_rate'],
                          mode='lines+markers', name=strategy.upper(),
                          line=dict(color=colors[i % len(colors)]),
                          showlegend=False),
                row=1, col=2
            )
        
        # 3. ドローダウン
        for i, strategy in enumerate(strategies):
            data = self.performance_df[self.performance_df['strategy'] == strategy]
            fig.add_trace(
                go.Scatter(x=data['date'], y=-data['max_drawdown'] * 100,
                          mode='lines', name=strategy.upper(),
                          fill='tozeroy',
                          line=dict(color=colors[i % len(colors)]),
                          showlegend=False),
                row=2, col=1
            )
        
        # 4. 取引分布（パイチャート）
        if not self.trades_df.empty:
            trade_counts = self.trades_df['strategy'].value_counts()
            fig.add_trace(
                go.Pie(labels=[s.upper() for s in trade_counts.index],
                      values=trade_counts.values,
                      hole=.3),
                row=2, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            title_text="自動ペーパートレーディング総合ダッシュボード",
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        # 軸ラベル
        fig.update_xaxes(title_text="日付", row=1, col=1)
        fig.update_yaxes(title_text="累積損益 ($)", row=1, col=1)
        fig.update_xaxes(title_text="日付", row=1, col=2)
        fig.update_yaxes(title_text="勝率 (%)", row=1, col=2)
        fig.update_xaxes(title_text="日付", row=2, col=1)
        fig.update_yaxes(title_text="ドローダウン (%)", row=2, col=1)
        
        # HTMLファイル保存
        fig.write_html(self.results_dir / 'interactive_dashboard.html')
    
    def generate_summary_report(self):
        """サマリーレポート生成"""
        print("\n📝 サマリーレポート生成...")
        
        report_path = self.results_dir / f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("自動ペーパートレーディング分析レポート\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not self.performance_df.empty:
                # 戦略ランキング
                final_performance = self.performance_df.groupby('strategy').agg({
                    'total_pnl': 'last',
                    'win_rate': 'mean',
                    'sharpe_ratio': 'mean',
                    'max_drawdown': 'max'
                }).round(2)
                
                final_performance = final_performance.sort_values('total_pnl', ascending=False)
                
                f.write("🏆 戦略ランキング（総損益順）:\n\n")
                for i, (strategy, row) in enumerate(final_performance.iterrows(), 1):
                    f.write(f"{i}. {strategy.upper()}\n")
                    f.write(f"   総損益: ${row['total_pnl']:,.2f}\n")
                    f.write(f"   平均勝率: {row['win_rate']:.1f}%\n")
                    f.write(f"   平均シャープレシオ: {row['sharpe_ratio']:.2f}\n")
                    f.write(f"   最大ドローダウン: {row['max_drawdown']*100:.1f}%\n\n")
                
                # 推奨事項
                f.write("\n💡 推奨事項:\n\n")
                
                best_strategy = final_performance.index[0]
                f.write(f"1. 最も収益性の高い戦略: {best_strategy.upper()}\n")
                
                # シャープレシオが最も高い戦略
                best_sharpe = final_performance.sort_values('sharpe_ratio', ascending=False).index[0]
                f.write(f"2. リスク調整後リターンが最も良い戦略: {best_sharpe.upper()}\n")
                
                # 最も安定した戦略（低ドローダウン）
                most_stable = final_performance.sort_values('max_drawdown').index[0]
                f.write(f"3. 最も安定した戦略: {most_stable.upper()}\n")
        
        print(f"✅ サマリーレポート保存: {report_path}")

def main():
    """メイン関数"""
    analyzer = AutomatedTestAnalyzer()
    
    print("🔍 自動テスト結果分析開始")
    
    # 包括的レポート生成
    analyzer.generate_comprehensive_report()
    
    # サマリーレポート生成
    analyzer.generate_summary_report()
    
    print("\n✨ 分析完了！")

if __name__ == "__main__":
    main()