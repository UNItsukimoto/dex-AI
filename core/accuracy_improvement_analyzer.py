#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
予測精度改善分析ツール
未達成期間の詳細分析と改善策の特定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 改善システムをインポート
from improved_prediction_system import ImprovedPredictionSystem

class AccuracyImprovementAnalyzer:
    """予測精度改善分析クラス"""
    
    def __init__(self):
        self.system = ImprovedPredictionSystem()
        self.results_dir = Path("results/accuracy_improvement")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 目標未達成期間
        self.target_periods = ['current', '2025_06', '2025_05']
        self.current_accuracies = {
            'current': 0.4839,
            '2025_06': 0.4833,
            '2025_05': 0.4885
        }
        
    def analyze_underperforming_periods(self):
        """未達成期間の詳細分析"""
        logger.info("未達成期間の詳細分析を開始...")
        
        analysis_results = {}
        
        for period in self.target_periods:
            logger.info(f"\n=== {period} 期間の分析 ===")
            
            # データ読み込み
            df = self.system.load_period_data(period)
            if df is None or len(df) < 30:
                logger.warning(f"{period}: データ不足")
                continue
            
            # 分析実行
            period_analysis = self._analyze_single_period(period, df)
            analysis_results[period] = period_analysis
        
        return analysis_results
    
    def _analyze_single_period(self, period_name, df):
        """単一期間の詳細分析"""
        logger.info(f"{period_name} の分析開始 - データ数: {len(df)}")
        
        # 特徴量作成
        df_features = self.system.create_advanced_features(df)
        
        # 基本統計
        basic_stats = {
            'data_points': len(df),
            'feature_points': len(df_features),
            'data_loss_ratio': (len(df) - len(df_features)) / len(df),
            'price_volatility': df['close'].pct_change().std(),
            'price_trend': (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0],
            'volume_stability': df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 0
        }
        
        # 特徴量分析
        exclude_cols = ['target', 'target_direction', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # 欠損値処理
        X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
        
        # 特徴量の統計
        feature_stats = {
            'total_features': len(feature_cols),
            'nan_ratio': np.isnan(X).sum() / X.size,
            'inf_ratio': np.isinf(X).sum() / X.size,
            'zero_variance_features': np.sum(np.var(X, axis=0) == 0),
            'correlation_with_target': self._calculate_feature_correlations(X, y, feature_cols)
        }
        
        # 予測性能分析
        performance_analysis = self._analyze_prediction_performance(X, y, period_name)
        
        # 問題特定
        issues = self._identify_issues(basic_stats, feature_stats, performance_analysis)
        
        return {
            'period': period_name,
            'current_accuracy': self.current_accuracies.get(period_name, 0),
            'target_accuracy': 0.50,
            'improvement_needed': 0.50 - self.current_accuracies.get(period_name, 0),
            'basic_stats': basic_stats,
            'feature_stats': feature_stats,
            'performance_analysis': performance_analysis,
            'identified_issues': issues,
            'recommendations': self._generate_recommendations(issues)
        }
    
    def _calculate_feature_correlations(self, X, y, feature_names):
        """特徴量とターゲットの相関分析"""
        correlations = []
        
        for i, feature_name in enumerate(feature_names):
            try:
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(correlation):
                    correlations.append({
                        'feature': feature_name,
                        'correlation': abs(correlation),
                        'raw_correlation': correlation
                    })
            except:
                continue
        
        # 相関の高い特徴量トップ10
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return {
            'top_10_features': correlations[:10],
            'avg_correlation': np.mean([c['correlation'] for c in correlations]) if correlations else 0,
            'high_correlation_count': len([c for c in correlations if c['correlation'] > 0.1])
        }
    
    def _analyze_prediction_performance(self, X, y, period_name):
        """予測性能の詳細分析"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import RobustScaler
        
        # データ分割
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 単純なRFモデルで性能確認
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            # 交差検証
            cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
            
            # 訓練・テスト
            rf.fit(X_train_scaled, y_train)
            train_pred = rf.predict(X_train_scaled)
            test_pred = rf.predict(X_test_scaled)
            
            # 方向性精度
            train_direction_acc = self._calculate_direction_accuracy(y_train, train_pred)
            test_direction_acc = self._calculate_direction_accuracy(y_test, test_pred)
            
            # 特徴量重要度
            feature_importance = rf.feature_importances_
            
            return {
                'cv_mse': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_direction_accuracy': train_direction_acc,
                'test_direction_accuracy': test_direction_acc,
                'overfitting_indicator': train_direction_acc - test_direction_acc,
                'prediction_variance': np.var(test_pred),
                'target_variance': np.var(y_test),
                'top_feature_importance': np.argsort(feature_importance)[-10:][::-1].tolist()
            }
            
        except Exception as e:
            logger.error(f"性能分析エラー: {e}")
            return {'error': str(e)}
    
    def _calculate_direction_accuracy(self, y_true, y_pred):
        """方向性精度計算"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        mask = true_direction != 0
        if np.sum(mask) == 0:
            return 0.5
        
        true_direction = true_direction[mask]
        pred_direction = pred_direction[mask]
        
        return np.mean(true_direction == pred_direction)
    
    def _identify_issues(self, basic_stats, feature_stats, performance_analysis):
        """問題点の特定"""
        issues = []
        
        # データ品質の問題
        if basic_stats['data_loss_ratio'] > 0.5:
            issues.append({
                'type': 'data_quality',
                'severity': 'high',
                'description': f"特徴量作成時のデータ損失が大きい ({basic_stats['data_loss_ratio']:.1%})",
                'impact': 'モデル訓練データ不足'
            })
        
        # 特徴量の問題
        if feature_stats['zero_variance_features'] > 5:
            issues.append({
                'type': 'feature_quality',
                'severity': 'medium',
                'description': f"分散がゼロの特徴量が多い ({feature_stats['zero_variance_features']}個)",
                'impact': '情報価値の低い特徴量'
            })
        
        # 相関の問題
        if feature_stats['correlation_with_target']['avg_correlation'] < 0.05:
            issues.append({
                'type': 'feature_relevance',
                'severity': 'high',
                'description': f"ターゲットとの平均相関が低い ({feature_stats['correlation_with_target']['avg_correlation']:.3f})",
                'impact': '予測力不足'
            })
        
        # オーバーフィッティング
        if 'overfitting_indicator' in performance_analysis and performance_analysis['overfitting_indicator'] > 0.1:
            issues.append({
                'type': 'overfitting',
                'severity': 'high',
                'description': f"オーバーフィッティングの兆候 (差: {performance_analysis['overfitting_indicator']:.3f})",
                'impact': '汎化性能の低下'
            })
        
        # ボラティリティの問題
        if basic_stats['price_volatility'] > 0.05:
            issues.append({
                'type': 'market_volatility',
                'severity': 'medium',
                'description': f"高ボラティリティ期間 ({basic_stats['price_volatility']:.3f})",
                'impact': '予測困難な市場状況'
            })
        
        return issues
    
    def _generate_recommendations(self, issues):
        """改善推奨事項の生成"""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'data_quality':
                recommendations.append({
                    'priority': 'high',
                    'action': '特徴量作成プロセスの最適化',
                    'details': [
                        '欠損値補完手法の改良',
                        'より短期間の移動平均使用',
                        '前方/後方補完の活用'
                    ]
                })
            
            elif issue['type'] == 'feature_relevance':
                recommendations.append({
                    'priority': 'high',
                    'action': '特徴量選択の改良',
                    'details': [
                        '相互情報量ベースの選択強化',
                        'ドメイン知識による特徴量追加',
                        '非線形特徴量の導入'
                    ]
                })
            
            elif issue['type'] == 'overfitting':
                recommendations.append({
                    'priority': 'high',
                    'action': '正則化の強化',
                    'details': [
                        'アンサンブル手法の調整',
                        'クロスバリデーション強化',
                        'ドロップアウト導入'
                    ]
                })
            
            elif issue['type'] == 'market_volatility':
                recommendations.append({
                    'priority': 'medium',
                    'action': 'ボラティリティ対応',
                    'details': [
                        'ボラティリティ調整された特徴量',
                        '市場レジーム別モデル',
                        'アダプティブ予測窓'
                    ]
                })
        
        # 基本的な改善策
        recommendations.append({
            'priority': 'medium',
            'action': 'ハイパーパラメータ最適化',
            'details': [
                'Bayesian Optimization',
                'Grid Search強化',
                'アンサンブル重み調整'
            ]
        })
        
        return recommendations
    
    def generate_improvement_plan(self, analysis_results):
        """改善計画の生成"""
        logger.info("改善計画を生成中...")
        
        plan = {
            'summary': {
                'target_periods': len(self.target_periods),
                'current_avg_accuracy': np.mean(list(self.current_accuracies.values())),
                'target_accuracy': 0.50,
                'total_improvement_needed': sum([0.50 - acc for acc in self.current_accuracies.values()])
            },
            'period_priorities': [],
            'technical_actions': [],
            'implementation_order': []
        }
        
        # 期間別優先度
        for period, result in analysis_results.items():
            improvement_needed = result['improvement_needed']
            issue_severity = len([i for i in result['identified_issues'] if i['severity'] == 'high'])
            
            plan['period_priorities'].append({
                'period': period,
                'current_accuracy': result['current_accuracy'],
                'improvement_needed': improvement_needed,
                'issue_count': len(result['identified_issues']),
                'priority_score': improvement_needed + (issue_severity * 0.01)
            })
        
        # 優先度でソート
        plan['period_priorities'].sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 技術的アクション
        all_issues = []
        for result in analysis_results.values():
            all_issues.extend(result['identified_issues'])
        
        # 問題タイプ別の統計
        issue_types = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1
        
        # 最も頻出する問題から対処
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            plan['technical_actions'].append({
                'issue_type': issue_type,
                'frequency': count,
                'priority': 'high' if count >= 2 else 'medium'
            })
        
        # 実装順序
        plan['implementation_order'] = [
            "1. データ品質向上（欠損値処理改善）",
            "2. 特徴量選択アルゴリズム改良", 
            "3. 正則化パラメータ調整",
            "4. アンサンブル重み最適化",
            "5. 期間別モデル微調整"
        ]
        
        return plan
    
    def save_results(self, analysis_results, improvement_plan):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON保存
        with open(self.results_dir / f'analysis_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        with open(self.results_dir / f'improvement_plan_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(improvement_plan, f, indent=2, ensure_ascii=False, default=str)
        
        # レポート生成
        self._generate_report(analysis_results, improvement_plan)
        
        logger.info(f"結果保存完了: {self.results_dir}")
    
    def _generate_report(self, analysis_results, improvement_plan):
        """改善レポート生成"""
        report = f"""
# 予測精度改善分析レポート

実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 概要

### 目標
- 全期間で予測精度50%以上達成
- 現在未達成: {len(self.target_periods)}期間

### 現在の状況
"""
        
        for period in self.target_periods:
            acc = self.current_accuracies[period]
            needed = 0.50 - acc
            report += f"- {period}: {acc:.2%} (改善必要: +{needed:.2%})\n"
        
        report += f"""
### 優先度ランキング
"""
        
        for i, period_priority in enumerate(improvement_plan['period_priorities'], 1):
            period = period_priority['period']
            score = period_priority['priority_score']
            report += f"{i}. {period} (優先度スコア: {score:.3f})\n"
        
        report += f"""

## 期間別分析結果

"""
        
        for period, result in analysis_results.items():
            report += f"### {period} 期間\n\n"
            report += f"- 現在精度: {result['current_accuracy']:.2%}\n"
            report += f"- 改善必要: +{result['improvement_needed']:.2%}\n"
            report += f"- データ数: {result['basic_stats']['data_points']}\n"
            report += f"- 特徴量数: {result['feature_stats']['total_features']}\n"
            
            report += f"\n**特定された問題:**\n"
            for issue in result['identified_issues']:
                report += f"- [{issue['severity'].upper()}] {issue['description']}\n"
            
            report += f"\n**推奨改善策:**\n"
            for rec in result['recommendations']:
                report += f"- [{rec['priority'].upper()}] {rec['action']}\n"
                for detail in rec['details']:
                    report += f"  - {detail}\n"
            report += "\n"
        
        report += f"""

## 実装計画

### 技術的アクション（優先順）
"""
        
        for action in improvement_plan['technical_actions']:
            report += f"- {action['issue_type']} (頻度: {action['frequency']}, 優先度: {action['priority']})\n"
        
        report += f"""

### 実装順序
"""
        
        for order in improvement_plan['implementation_order']:
            report += f"{order}\n"
        
        report += f"""

## 次のステップ

1. **最優先**: {improvement_plan['period_priorities'][0]['period']} 期間の改善
2. **技術改善**: データ品質向上と特徴量選択改良
3. **検証**: 改善後の性能測定

---
*このレポートは自動生成されました*
"""
        
        with open(self.results_dir / 'improvement_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    """メイン実行"""
    analyzer = AccuracyImprovementAnalyzer()
    
    logger.info("🎯 予測精度改善分析を開始...")
    
    try:
        # 未達成期間の分析
        analysis_results = analyzer.analyze_underperforming_periods()
        
        if not analysis_results:
            logger.error("分析対象データが見つかりません")
            return
        
        # 改善計画生成
        improvement_plan = analyzer.generate_improvement_plan(analysis_results)
        
        # 結果保存
        analyzer.save_results(analysis_results, improvement_plan)
        
        # 結果表示
        print(f"\n{'='*60}")
        print("予測精度改善分析 完了")
        print(f"{'='*60}")
        
        print(f"\n分析対象期間: {len(analysis_results)}")
        for period, result in analysis_results.items():
            improvement = result['improvement_needed']
            print(f"  {period}: {result['current_accuracy']:.2%} → 50.0% (+{improvement:.2%})")
        
        print(f"\n最優先改善期間: {improvement_plan['period_priorities'][0]['period']}")
        
        print(f"\n主要な問題:")
        for action in improvement_plan['technical_actions'][:3]:
            print(f"  - {action['issue_type']} (頻度: {action['frequency']})")
        
        print(f"\n詳細レポート: results/accuracy_improvement/improvement_report.md")
        
    except Exception as e:
        logger.error(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()