#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リスク管理システム
高度なリスク制御とポジション管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """リスクレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class AlertType(Enum):
    """アラートタイプ"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"

@dataclass
class RiskMetrics:
    """リスク指標"""
    total_exposure: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk (95%)
    portfolio_volatility: float
    concentration_risk: float
    leverage_ratio: float
    risk_level: RiskLevel

@dataclass
class RiskLimit:
    """リスク制限"""
    max_position_size: float = 0.1  # 最大ポジションサイズ (10%)
    max_total_exposure: float = 0.5  # 最大総エクスポージャー (50%)
    max_drawdown: float = 0.15  # 最大ドローダウン (15%)
    max_daily_loss: float = 0.05  # 最大日次損失 (5%)
    max_leverage: float = 2.0  # 最大レバレッジ
    min_sharpe_ratio: float = 0.5  # 最小シャープレシオ
    correlation_threshold: float = 0.7  # 相関閾値

@dataclass
class PositionSizeRecommendation:
    """ポジションサイズ推奨"""
    symbol: str
    recommended_size: float
    max_allowed_size: float
    risk_adjusted_size: float
    confidence_multiplier: float
    volatility_adjustment: float
    reason: str

class RiskManagementSystem:
    """高度リスク管理システム"""
    
    def __init__(self, risk_limits: Optional[RiskLimit] = None):
        self.risk_limits = risk_limits or RiskLimit()
        self.price_history: Dict[str, List[Dict]] = {}
        self.portfolio_history: List[Dict] = []
        self.risk_alerts: List[Dict] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Kelly Criterion設定
        self.kelly_lookback_days = 30
        self.kelly_max_fraction = 0.25  # 最大25%
        
        logger.info("Risk Management System initialized")
    
    def update_price_history(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """価格履歴更新"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': timestamp,
            'price': price
        })
        
        # 履歴サイズ制限（過去1000ポイント）
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def update_portfolio_history(self, equity: float, positions: Dict, timestamp: Optional[datetime] = None):
        """ポートフォリオ履歴更新"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'positions': positions.copy()
        })
        
        # 履歴サイズ制限
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def calculate_volatility(self, symbol: str, days: int = 30) -> float:
        """ボラティリティ計算"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.5  # デフォルト値
        
        prices = [p['price'] for p in self.price_history[symbol][-days*24:]]  # 時間足想定
        if len(prices) < 2:
            return 0.5
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(24 * 365)  # 年率ボラティリティ
    
    def calculate_correlation_matrix(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """相関行列計算"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) >= 2:
                    prices = [p['price'] for p in self.price_history[symbol][-days*24:]]
                    if len(prices) >= 2:
                        returns = np.diff(np.log(prices))
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # データ長を揃える
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_data = {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
            
            df = pd.DataFrame(aligned_data)
            correlation_matrix = df.corr()
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation calculation error: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_var(self, positions: Dict, confidence: float = 0.95) -> float:
        """ポートフォリオVaR計算"""
        try:
            if not positions:
                return 0.0
            
            symbols = list(positions.keys())
            weights = []
            volatilities = []
            
            total_value = sum(abs(pos['quantity'] * pos['current_price']) for pos in positions.values())
            
            for symbol in symbols:
                pos = positions[symbol]
                position_value = abs(pos['quantity'] * pos['current_price'])
                weight = position_value / total_value if total_value > 0 else 0
                volatility = self.calculate_volatility(symbol)
                
                weights.append(weight)
                volatilities.append(volatility)
            
            # 相関行列取得
            corr_matrix = self.calculate_correlation_matrix(symbols)
            
            if corr_matrix.empty:
                # 相関データがない場合は保守的に計算
                portfolio_variance = sum(w * v for w, v in zip(weights, volatilities)) ** 2
            else:
                # 正確なポートフォリオ分散計算
                weights_array = np.array(weights)
                vol_array = np.array(volatilities)
                
                # 分散共分散行列
                cov_matrix = np.outer(vol_array, vol_array) * corr_matrix.values
                portfolio_variance = np.dot(weights_array, np.dot(cov_matrix, weights_array))
            
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # VaR計算（正規分布仮定）
            z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99%
            var = portfolio_volatility * z_score
            
            return var
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.1  # 保守的なデフォルト値
    
    def calculate_kelly_criterion(self, symbol: str, win_probability: float, 
                                 avg_win: float, avg_loss: float) -> float:
        """Kelly基準によるポジションサイズ計算"""
        try:
            if avg_loss <= 0:
                return 0.0
            
            # Kelly fraction = (bp - q) / b
            # b = avg_win / avg_loss (勝率)
            # p = win_probability
            # q = 1 - p (負率)
            
            b = avg_win / abs(avg_loss)
            p = win_probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # 制限を適用
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_max_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly criterion calculation error: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, account_equity: float, 
                               prediction_confidence: float, current_price: float,
                               historical_performance: Optional[Dict] = None) -> PositionSizeRecommendation:
        """高度ポジションサイズ計算"""
        try:
            # 1. 基本ポジションサイズ（資金の固定%）
            base_size_pct = 0.05  # 5%
            base_size = account_equity * base_size_pct
            base_quantity = base_size / current_price
            
            # 2. 信頼度調整
            confidence_multiplier = min(2.0, max(0.5, prediction_confidence / 0.5))
            
            # 3. ボラティリティ調整
            volatility = self.calculate_volatility(symbol)
            vol_adjustment = 1.0 / (1.0 + volatility)  # 高ボラティリティで減額
            
            # 4. Kelly基準調整（履歴がある場合）
            kelly_multiplier = 1.0
            if historical_performance:
                win_rate = historical_performance.get('win_rate', 0.5)
                avg_win = historical_performance.get('avg_win', 0.02)
                avg_loss = historical_performance.get('avg_loss', -0.02)
                
                kelly_fraction = self.calculate_kelly_criterion(symbol, win_rate, avg_win, avg_loss)
                kelly_multiplier = kelly_fraction / base_size_pct if base_size_pct > 0 else 1.0
                kelly_multiplier = max(0.2, min(3.0, kelly_multiplier))
            
            # 5. 相関調整（同方向ポジション過多を防ぐ）
            correlation_adjustment = 1.0
            if self.correlation_matrix is not None and symbol in self.correlation_matrix.columns:
                # 他ポジションとの相関を考慮
                high_corr_count = sum(1 for corr in self.correlation_matrix[symbol] 
                                    if abs(corr) > self.risk_limits.correlation_threshold)
                if high_corr_count > 1:
                    correlation_adjustment = 0.7  # 高相関時は減額
            
            # 6. 最終ポジションサイズ計算
            risk_adjusted_quantity = (base_quantity * confidence_multiplier * 
                                    vol_adjustment * kelly_multiplier * correlation_adjustment)
            
            # 7. 制限チェック
            max_position_value = account_equity * self.risk_limits.max_position_size
            max_allowed_quantity = max_position_value / current_price
            
            recommended_quantity = min(risk_adjusted_quantity, max_allowed_quantity)
            
            # 8. 最小取引単位調整
            if symbol == 'BTC':
                recommended_quantity = round(recommended_quantity, 3)
            elif symbol == 'ETH':
                recommended_quantity = round(recommended_quantity, 2)
            else:
                recommended_quantity = round(recommended_quantity, 1)
            
            # 最小取引額チェック（$10以上）
            if recommended_quantity * current_price < 10:
                recommended_quantity = 0
            
            reason_parts = []
            if confidence_multiplier != 1.0:
                reason_parts.append(f"信頼度調整×{confidence_multiplier:.2f}")
            if vol_adjustment < 0.9:
                reason_parts.append(f"ボラティリティ調整×{vol_adjustment:.2f}")
            if kelly_multiplier != 1.0:
                reason_parts.append(f"Kelly調整×{kelly_multiplier:.2f}")
            if correlation_adjustment < 1.0:
                reason_parts.append("相関リスク減額")
            
            reason = "; ".join(reason_parts) if reason_parts else "標準計算"
            
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_size=recommended_quantity,
                max_allowed_size=max_allowed_quantity,
                risk_adjusted_size=risk_adjusted_quantity,
                confidence_multiplier=confidence_multiplier,
                volatility_adjustment=vol_adjustment,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_size=0,
                max_allowed_size=0,
                risk_adjusted_size=0,
                confidence_multiplier=1.0,
                volatility_adjustment=1.0,
                reason=f"計算エラー: {e}"
            )
    
    def calculate_stop_loss_take_profit(self, symbol: str, entry_price: float, 
                                       position_side: str, confidence: float) -> Dict:
        """ストップロス・テイクプロフィット計算"""
        try:
            volatility = self.calculate_volatility(symbol)
            
            # 基本設定
            base_stop_loss = 0.02  # 2%
            base_take_profit = 0.04  # 4%
            
            # ボラティリティ調整
            vol_multiplier = 1 + volatility  # 高ボラティリティで幅を広げる
            
            # 信頼度調整
            confidence_adjustment = 1.0 + (confidence - 0.5)  # 高信頼度で利確を遠く
            
            stop_loss_pct = base_stop_loss * vol_multiplier
            take_profit_pct = base_take_profit * vol_multiplier * confidence_adjustment
            
            if position_side.lower() == 'long':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # short
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            # 動的調整機能
            time_based_adjustment = self._get_time_based_adjustment()
            
            return {
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'volatility_factor': vol_multiplier,
                'confidence_factor': confidence_adjustment,
                'time_adjustment': time_based_adjustment,
                'trailing_stop_enabled': confidence > 0.7
            }
            
        except Exception as e:
            logger.error(f"Stop loss/take profit calculation error: {e}")
            return {
                'stop_loss_price': entry_price * 0.98,  # デフォルト2%
                'take_profit_price': entry_price * 1.04,  # デフォルト4%
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'volatility_factor': 1.0,
                'confidence_factor': 1.0,
                'time_adjustment': 1.0,
                'trailing_stop_enabled': False
            }
    
    def _get_time_based_adjustment(self) -> float:
        """時間ベース調整係数"""
        current_hour = datetime.now().hour
        
        # 市場時間に基づく調整
        if 8 <= current_hour <= 16:  # 通常時間
            return 1.0
        elif 16 < current_hour <= 20:  # 夕方（ボラティリティ高）
            return 1.2
        else:  # 夜間・早朝（流動性低）
            return 0.8
    
    def calculate_risk_metrics(self, account_equity: float, positions: Dict) -> RiskMetrics:
        """リスク指標計算"""
        try:
            # 1. 総エクスポージャー
            total_exposure = sum(abs(pos['quantity'] * pos['current_price']) 
                               for pos in positions.values()) / account_equity
            
            # 2. 最大ドローダウン
            max_drawdown = self._calculate_max_drawdown()
            
            # 3. シャープレシオ
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # 4. VaR
            var_95 = self.calculate_portfolio_var(positions, 0.95)
            
            # 5. ポートフォリオボラティリティ
            portfolio_volatility = self._calculate_portfolio_volatility(positions)
            
            # 6. 集中リスク
            concentration_risk = self._calculate_concentration_risk(positions, account_equity)
            
            # 7. レバレッジ比率
            leverage_ratio = total_exposure
            
            # 8. リスクレベル判定
            risk_level = self._determine_risk_level(total_exposure, max_drawdown, 
                                                  var_95, concentration_risk)
            
            return RiskMetrics(
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                portfolio_volatility=portfolio_volatility,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return RiskMetrics(
                total_exposure=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                var_95=0.0,
                portfolio_volatility=0.0,
                concentration_risk=0.0,
                leverage_ratio=0.0,
                risk_level=RiskLevel.LOW
            )
    
    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウン計算"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        equity_values = [p['equity'] for p in self.portfolio_history]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """シャープレシオ計算"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        equity_values = [p['equity'] for p in self.portfolio_history]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252  # 日次リスクフリーレート
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        return (excess_returns / volatility) * np.sqrt(252)  # 年率化
    
    def _calculate_portfolio_volatility(self, positions: Dict) -> float:
        """ポートフォリオボラティリティ計算"""
        if not positions:
            return 0.0
        
        symbols = list(positions.keys())
        volatilities = [self.calculate_volatility(symbol) for symbol in symbols]
        
        total_value = sum(abs(pos['quantity'] * pos['current_price']) for pos in positions.values())
        weights = [abs(pos['quantity'] * pos['current_price']) / total_value 
                  for pos in positions.values()]
        
        # 単純加重平均（相関考慮なし）
        weighted_vol = sum(w * v for w, v in zip(weights, volatilities))
        
        return weighted_vol
    
    def _calculate_concentration_risk(self, positions: Dict, account_equity: float) -> float:
        """集中リスク計算"""
        if not positions:
            return 0.0
        
        position_values = [abs(pos['quantity'] * pos['current_price']) for pos in positions.values()]
        total_value = sum(position_values)
        
        if total_value == 0:
            return 0.0
        
        # 最大ポジションの比率
        max_position = max(position_values)
        concentration = max_position / account_equity
        
        return concentration
    
    def _determine_risk_level(self, exposure: float, drawdown: float, 
                            var: float, concentration: float) -> RiskLevel:
        """リスクレベル判定"""
        risk_score = 0
        
        # エクスポージャー
        if exposure > 0.8:
            risk_score += 3
        elif exposure > 0.5:
            risk_score += 2
        elif exposure > 0.3:
            risk_score += 1
        
        # ドローダウン
        if drawdown > 0.2:
            risk_score += 3
        elif drawdown > 0.1:
            risk_score += 2
        elif drawdown > 0.05:
            risk_score += 1
        
        # VaR
        if var > 0.15:
            risk_score += 2
        elif var > 0.1:
            risk_score += 1
        
        # 集中リスク
        if concentration > 0.2:
            risk_score += 2
        elif concentration > 0.1:
            risk_score += 1
        
        if risk_score >= 7:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_risk_violations(self, account_equity: float, positions: Dict) -> List[Dict]:
        """リスク制限違反チェック"""
        violations = []
        
        try:
            metrics = self.calculate_risk_metrics(account_equity, positions)
            
            # エクスポージャーチェック
            if metrics.total_exposure > self.risk_limits.max_total_exposure:
                violations.append({
                    'type': AlertType.POSITION_SIZE.value,
                    'severity': 'high',
                    'message': f"総エクスポージャー超過: {metrics.total_exposure:.1%} > {self.risk_limits.max_total_exposure:.1%}",
                    'current_value': metrics.total_exposure,
                    'limit': self.risk_limits.max_total_exposure
                })
            
            # ドローダウンチェック
            if metrics.max_drawdown > self.risk_limits.max_drawdown:
                violations.append({
                    'type': AlertType.DRAWDOWN.value,
                    'severity': 'high',
                    'message': f"最大ドローダウン超過: {metrics.max_drawdown:.1%} > {self.risk_limits.max_drawdown:.1%}",
                    'current_value': metrics.max_drawdown,
                    'limit': self.risk_limits.max_drawdown
                })
            
            # ボラティリティチェック
            if metrics.portfolio_volatility > 1.0:  # 100%
                violations.append({
                    'type': AlertType.VOLATILITY.value,
                    'severity': 'medium',
                    'message': f"ポートフォリオボラティリティ高: {metrics.portfolio_volatility:.1%}",
                    'current_value': metrics.portfolio_volatility,
                    'limit': 1.0
                })
            
            # 集中リスクチェック
            if metrics.concentration_risk > self.risk_limits.max_position_size:
                violations.append({
                    'type': AlertType.POSITION_SIZE.value,
                    'severity': 'medium',
                    'message': f"集中リスク: {metrics.concentration_risk:.1%} > {self.risk_limits.max_position_size:.1%}",
                    'current_value': metrics.concentration_risk,
                    'limit': self.risk_limits.max_position_size
                })
            
        except Exception as e:
            logger.error(f"Risk violation check error: {e}")
        
        return violations
    
    def generate_risk_report(self, account_equity: float, positions: Dict) -> Dict:
        """リスクレポート生成"""
        try:
            metrics = self.calculate_risk_metrics(account_equity, positions)
            violations = self.check_risk_violations(account_equity, positions)
            
            # ポジション別詳細
            position_details = []
            for symbol, pos in positions.items():
                position_value = abs(pos['quantity'] * pos['current_price'])
                position_pct = position_value / account_equity
                volatility = self.calculate_volatility(symbol)
                
                position_details.append({
                    'symbol': symbol,
                    'value': position_value,
                    'percentage': position_pct,
                    'volatility': volatility,
                    'risk_contribution': position_pct * volatility
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'account_equity': account_equity,
                'risk_metrics': {
                    'total_exposure': metrics.total_exposure,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'var_95': metrics.var_95,
                    'portfolio_volatility': metrics.portfolio_volatility,
                    'concentration_risk': metrics.concentration_risk,
                    'leverage_ratio': metrics.leverage_ratio,
                    'risk_level': metrics.risk_level.value
                },
                'violations': violations,
                'position_details': position_details,
                'recommendations': self._generate_recommendations(metrics, violations)
            }
            
        except Exception as e:
            logger.error(f"Risk report generation error: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_recommendations(self, metrics: RiskMetrics, violations: List[Dict]) -> List[str]:
        """リスク改善推奨事項生成"""
        recommendations = []
        
        if metrics.risk_level == RiskLevel.EXTREME:
            recommendations.append("緊急: リスクレベルが極度に高い状態です。ポジションの大幅削減を検討してください。")
        elif metrics.risk_level == RiskLevel.HIGH:
            recommendations.append("警告: リスクレベルが高い状態です。ポジションサイズの見直しを推奨します。")
        
        if metrics.total_exposure > 0.7:
            recommendations.append("総エクスポージャーが高すぎます。一部ポジションの決済を検討してください。")
        
        if metrics.concentration_risk > 0.15:
            recommendations.append("特定銘柄への集中が見られます。ポートフォリオの分散を図ってください。")
        
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("リスク調整後リターンが低い状態です。取引戦略の見直しを推奨します。")
        
        if len(violations) > 0:
            recommendations.append(f"{len(violations)}件のリスク制限違反が検出されました。詳細を確認してください。")
        
        if not recommendations:
            recommendations.append("現在のリスクレベルは適切な範囲内です。")
        
        return recommendations

# テスト関数
def test_risk_management():
    """リスク管理システムのテスト"""
    print("=== Risk Management System Test ===")
    
    # システム初期化
    risk_mgmt = RiskManagementSystem()
    
    # 価格履歴追加
    symbols = ['BTC', 'ETH', 'SOL']
    base_prices = {'BTC': 67000, 'ETH': 3200, 'SOL': 180}
    
    for i in range(100):
        timestamp = datetime.now() - timedelta(hours=100-i)
        for symbol in symbols:
            # 価格変動シミュレーション
            price = base_prices[symbol] * (1 + np.random.normal(0, 0.02))
            risk_mgmt.update_price_history(symbol, price, timestamp)
    
    # ポジション設定
    positions = {
        'BTC': {
            'quantity': 0.15,
            'current_price': 67000,
            'entry_price': 66000,
            'unrealized_pnl': 150
        },
        'ETH': {
            'quantity': 3.0,
            'current_price': 3200,
            'entry_price': 3100,
            'unrealized_pnl': 300
        }
    }
    
    account_equity = 10000
    
    # ポートフォリオ履歴追加
    risk_mgmt.update_portfolio_history(account_equity, positions)
    
    print("1. Position Size Calculation:")
    for symbol in symbols:
        recommendation = risk_mgmt.calculate_position_size(
            symbol, account_equity, 0.7, base_prices[symbol]
        )
        print(f"   {symbol}: {recommendation.recommended_size:.3f} (理由: {recommendation.reason})")
    
    print("\n2. Stop Loss / Take Profit:")
    for symbol in ['BTC', 'ETH']:
        sl_tp = risk_mgmt.calculate_stop_loss_take_profit(
            symbol, base_prices[symbol], 'long', 0.7
        )
        print(f"   {symbol}: SL=${sl_tp['stop_loss_price']:.0f} TP=${sl_tp['take_profit_price']:.0f}")
    
    print("\n3. Risk Metrics:")
    metrics = risk_mgmt.calculate_risk_metrics(account_equity, positions)
    print(f"   Total Exposure: {metrics.total_exposure:.1%}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.1%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   VaR 95%: {metrics.var_95:.1%}")
    print(f"   Risk Level: {metrics.risk_level.value}")
    
    print("\n4. Risk Violations:")
    violations = risk_mgmt.check_risk_violations(account_equity, positions)
    if violations:
        for violation in violations:
            print(f"   {violation['severity'].upper()}: {violation['message']}")
    else:
        print("   No violations detected")
    
    print("\n5. Risk Report:")
    report = risk_mgmt.generate_risk_report(account_equity, positions)
    recommendations = report.get('recommendations', [])
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    test_risk_management()