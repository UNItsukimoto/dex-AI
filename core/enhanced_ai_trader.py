#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
強化AIトレーダー
リスク管理統合版
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
import requests
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from paper_trading_engine import PaperTradingEngine, OrderSide, OrderType
from risk_management_system import RiskManagementSystem, RiskLimit
from advanced_prediction_engine import AdvancedPredictionEngine
from alert_notification_system import AlertNotificationSystem
from performance_analyzer import PerformanceAnalyzer
from multi_symbol_manager import MultiSymbolManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAITrader:
    """リスク管理統合AIトレーダー"""
    
    def __init__(self, initial_balance: float = 10000.0):
        # コア システム初期化
        self.trading_engine = PaperTradingEngine(initial_balance)
        
        # リスク管理設定
        risk_limits = RiskLimit(
            max_position_size=0.15,  # 15%
            max_total_exposure=0.6,  # 60%
            max_drawdown=0.2,       # 20%
            max_daily_loss=0.08,    # 8%
            max_leverage=3.0,       # 3倍
            correlation_threshold=0.6
        )
        self.risk_manager = RiskManagementSystem(risk_limits)
        
        # 高度予測エンジン初期化
        self.prediction_engine = AdvancedPredictionEngine()
        
        # アラート・通知システム初期化
        self.alert_system = AlertNotificationSystem()
        
        # パフォーマンス分析システム初期化
        self.performance_analyzer = PerformanceAnalyzer()
        
        # マルチ銘柄管理システム初期化
        self.multi_symbol_manager = MultiSymbolManager()
        
        # API設定
        self.api_base_url = "https://api.hyperliquid.xyz"
        
        # 取引設定（マルチ銘柄管理から取得）
        self.symbols = self.multi_symbol_manager.get_enabled_symbols()
        self.confidence_threshold = 0.5  # 高度予測で厳格化
        self.buy_threshold = 0.68       # 高度予測で厳格化  
        self.sell_threshold = 0.32      # 高度予測で厳格化
        
        # 履歴データ
        self.prediction_history: List[Dict] = []
        self.trade_signals: List[Dict] = []
        self.risk_reports: List[Dict] = []
        
        # パフォーマンス追跡
        self.performance_stats = {
            'total_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'avg_hold_time': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'streak_type': 'none'  # 'win', 'loss', 'none'
        }
        
        logger.info(f"Enhanced AI Trader with Advanced ML Engine and Alert System initialized with ${initial_balance:,.2f}")
    
    def get_live_prices(self) -> Dict[str, float]:
        """ライブ価格取得"""
        try:
            payload = {"type": "allMids"}
            response = requests.post(f"{self.api_base_url}/info", json=payload, timeout=10)
            
            if response.status_code == 200:
                all_mids = response.json()
                prices = {}
                
                for symbol in self.symbols:
                    if symbol in all_mids:
                        prices[symbol] = float(all_mids[symbol])
                
                return prices
            else:
                logger.error(f"Price API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Get live prices error: {e}")
            return {}
    
    def get_enhanced_prediction(self, symbol: str) -> Dict:
        """高度予測エンジンを使用した強化予測システム"""
        try:
            # ローソク足データ取得
            df = self.get_candles_for_prediction(symbol, 100)  # より多くのデータで精度向上
            if df.empty:
                return self._get_default_prediction(symbol)
            
            # 高度予測エンジンによる予測
            advanced_prediction = self.prediction_engine.get_enhanced_prediction(symbol, df)
            
            # 予測結果に追加情報を付与
            current_price = df['close'].iloc[-1] if not df.empty else 0
            advanced_prediction.update({
                'symbol': symbol,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'engine_type': 'advanced_ml_ensemble'
            })
            
            return advanced_prediction
            
        except Exception as e:
            logger.error(f"Advanced prediction error for {symbol}: {e}")
            return self._get_default_prediction(symbol)
    
    def get_candles_for_prediction(self, symbol: str, count: int = 60) -> pd.DataFrame:
        """予測用ローソク足データ取得"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (count * 60 * 60 * 1000)
            
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": "1h",
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            
            response = requests.post(f"{self.api_base_url}/info", json=payload, timeout=15)
            
            if response.status_code == 200:
                candles = response.json()
                
                if not candles:
                    return pd.DataFrame()
                
                df_data = []
                for candle in candles:
                    if isinstance(candle, dict) and 't' in candle:
                        df_data.append({
                            'timestamp': pd.to_datetime(candle['t'], unit='ms'),
                            'open': float(candle['o']),
                            'high': float(candle['h']),
                            'low': float(candle['l']),
                            'close': float(candle['c']),
                            'volume': float(candle['v']) if 'v' in candle else 0.0
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Get candles error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _create_advanced_features(self, df: pd.DataFrame) -> Optional[Dict]:
        """高度特徴量作成"""
        try:
            if len(df) < 20:
                return None
            
            features = {}
            
            # 価格関連
            features['current_price'] = df['close'].iloc[-1]
            features['price_change_1h'] = df['close'].pct_change().iloc[-1]
            features['price_change_4h'] = df['close'].pct_change(4).iloc[-1]
            features['price_change_24h'] = df['close'].pct_change(24).iloc[-1] if len(df) >= 24 else 0
            
            # 移動平均
            for period in [7, 14, 30]:
                if len(df) >= period:
                    ma = df['close'].rolling(period).mean()
                    features[f'ma_{period}'] = ma.iloc[-1]
                    features[f'ma_ratio_{period}'] = df['close'].iloc[-1] / ma.iloc[-1]
                    features[f'ma_slope_{period}'] = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2] if len(ma) >= 2 else 0
            
            # テクニカル指標
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = macd.ewm(span=9).mean().iloc[-1]
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # ボラティリティ
            features['volatility_10'] = df['close'].rolling(10).std().iloc[-1] / features['current_price']
            features['volatility_30'] = df['close'].rolling(30).std().iloc[-1] / features['current_price'] if len(df) >= 30 else features['volatility_10']
            
            # ボリンジャーバンド
            bb_period = 20
            if len(df) >= bb_period:
                bb_ma = df['close'].rolling(bb_period).mean()
                bb_std = df['close'].rolling(bb_period).std()
                features['bb_upper'] = (bb_ma + 2 * bb_std).iloc[-1]
                features['bb_lower'] = (bb_ma - 2 * bb_std).iloc[-1]
                features['bb_position'] = (features['current_price'] - bb_ma.iloc[-1]) / bb_std.iloc[-1]
            
            # サポート・レジスタンス
            recent_highs = df['high'].rolling(20).max()
            recent_lows = df['low'].rolling(20).min()
            features['resistance_distance'] = (recent_highs.iloc[-1] - features['current_price']) / features['current_price']
            features['support_distance'] = (features['current_price'] - recent_lows.iloc[-1]) / features['current_price']
            
            # 出来高関連
            features['volume_current'] = df['volume'].iloc[-1]
            features['volume_avg'] = df['volume'].rolling(20).mean().iloc[-1]
            features['volume_ratio'] = features['volume_current'] / features['volume_avg'] if features['volume_avg'] > 0 else 1
            
            # 時間要素
            current_time = df.index[-1]
            features['hour'] = current_time.hour
            features['day_of_week'] = current_time.dayofweek
            features['is_weekend'] = current_time.dayofweek >= 5
            
            return features
            
        except Exception as e:
            logger.error(f"Advanced features creation error: {e}")
            return None
    
    def _analyze_trend(self, features: Dict) -> float:
        """トレンド分析"""
        score = 0.0
        
        # MA slope analysis
        if 'ma_slope_7' in features and 'ma_slope_14' in features:
            if features['ma_slope_7'] > 0 and features['ma_slope_14'] > 0:
                score += 0.15
            elif features['ma_slope_7'] < 0 and features['ma_slope_14'] < 0:
                score -= 0.15
        
        # MA alignment
        if all(f'ma_ratio_{p}' in features for p in [7, 14, 30]):
            if features['ma_ratio_7'] > features['ma_ratio_14'] > features['ma_ratio_30']:
                score += 0.10
            elif features['ma_ratio_7'] < features['ma_ratio_14'] < features['ma_ratio_30']:
                score -= 0.10
        
        return score
    
    def _analyze_momentum(self, features: Dict) -> float:
        """モメンタム分析"""
        score = 0.0
        
        # Price momentum
        if 'price_change_1h' in features:
            if features['price_change_1h'] > 0.02:
                score += 0.10
            elif features['price_change_1h'] < -0.02:
                score -= 0.10
            elif features['price_change_1h'] > 0:
                score += 0.05
            else:
                score -= 0.05
        
        # MACD momentum
        if 'macd_histogram' in features:
            if features['macd_histogram'] > 0:
                score += 0.05
            else:
                score -= 0.05
        
        # Multi-timeframe momentum
        if 'price_change_4h' in features and 'price_change_24h' in features:
            momentum_alignment = (
                (features['price_change_1h'] > 0) +
                (features['price_change_4h'] > 0) +
                (features['price_change_24h'] > 0)
            )
            if momentum_alignment >= 2:
                score += 0.05
            elif momentum_alignment <= 1:
                score -= 0.05
        
        return score
    
    def _analyze_mean_reversion(self, features: Dict) -> float:
        """平均回帰分析"""
        score = 0.0
        
        # RSI mean reversion
        if 'rsi' in features:
            if features['rsi'] < 30:
                score += 0.10  # Oversold
            elif features['rsi'] > 70:
                score -= 0.10  # Overbought
            elif 40 < features['rsi'] < 60:
                score += 0.02  # Neutral zone
        
        # Bollinger Bands position
        if 'bb_position' in features:
            if features['bb_position'] < -1.5:
                score += 0.05  # Below lower band
            elif features['bb_position'] > 1.5:
                score -= 0.05  # Above upper band
        
        return score
    
    def _analyze_volatility(self, features: Dict) -> float:
        """ボラティリティ分析"""
        score = 0.0
        
        if 'volatility_10' in features and 'volatility_30' in features:
            vol_ratio = features['volatility_10'] / features['volatility_30']
            
            # Volatility regime
            if vol_ratio > 1.5:
                score -= 0.08  # High volatility reduces confidence
            elif vol_ratio < 0.7:
                score += 0.08  # Low volatility increases confidence
            
            # Absolute volatility check
            if features['volatility_10'] > 0.05:  # 5% daily vol
                score -= 0.07
        
        return score
    
    def _analyze_volume(self, features: Dict) -> float:
        """出来高分析"""
        score = 0.0
        
        if 'volume_ratio' in features:
            if features['volume_ratio'] > 1.5:
                score += 0.05  # High volume confirms moves
            elif features['volume_ratio'] < 0.5:
                score -= 0.05  # Low volume reduces confidence
        
        return score
    
    def _analyze_market_structure(self, features: Dict) -> float:
        """市場構造分析"""
        score = 0.0
        
        # Support/Resistance proximity
        if 'support_distance' in features and 'resistance_distance' in features:
            if features['support_distance'] < 0.02:  # Near support
                score += 0.05
            elif features['resistance_distance'] < 0.02:  # Near resistance
                score -= 0.05
        
        return score
    
    def _analyze_time_factors(self, features: Dict) -> float:
        """時間要素分析"""
        score = 0.0
        
        if 'hour' in features:
            # Market hours adjustment
            if 8 <= features['hour'] <= 16:  # Active hours
                score += 0.02
            elif features['hour'] in [0, 1, 2, 6, 7]:  # Low activity
                score -= 0.03
        
        if 'is_weekend' in features and features['is_weekend']:
            score -= 0.02  # Weekend effect
        
        return score
    
    def _get_market_condition_factor(self, features: Dict) -> float:
        """市場状況ファクター"""
        factor = 1.0
        
        # Volatility adjustment
        if 'volatility_10' in features:
            if features['volatility_10'] > 0.08:  # Very high volatility
                factor *= 0.7
            elif features['volatility_10'] > 0.05:  # High volatility
                factor *= 0.85
        
        return factor
    
    def _get_default_prediction(self, symbol: str) -> Dict:
        """デフォルト予測"""
        return {
            'symbol': symbol,
            'probability': 0.5,
            'confidence': 0.1,
            'signal': 'HOLD',
            'features': {},
            'prediction_components': {}
        }
    
    def execute_enhanced_strategy(self):
        """強化取引戦略実行"""
        try:
            logger.info("Executing enhanced trading strategy with risk management...")
            
            # 現在価格取得
            current_prices = self.get_live_prices()
            if not current_prices:
                logger.warning("No price data available")
                return
            
            # 価格とポートフォリオ履歴更新
            for symbol, price in current_prices.items():
                self.risk_manager.update_price_history(symbol, price)
                
                # 価格変動チェック（前回から5%以上の変動でアラート）
                if hasattr(self, 'prev_prices') and symbol in self.prev_prices:
                    prev_price = self.prev_prices[symbol]
                    price_change = (price - prev_price) / prev_price
                    
                    if abs(price_change) >= 0.05:  # 5%以上の変動
                        alert = self.alert_system.create_price_alert(symbol, price, price_change)
                        self.alert_system.send_alert(alert)
            
            # 現在価格を保存
            self.prev_prices = current_prices.copy()
            
            account_summary = self.trading_engine.get_account_summary()
            positions = self.trading_engine.get_positions()
            
            self.risk_manager.update_portfolio_history(account_summary['equity'], positions)
            
            # リスク評価
            risk_metrics = self.risk_manager.calculate_risk_metrics(account_summary['equity'], positions)
            risk_violations = self.risk_manager.check_risk_violations(account_summary['equity'], positions)
            
            # リスク違反アラート送信
            for violation in risk_violations:
                if violation['severity'] == 'high':
                    alert = self.alert_system.create_risk_warning_alert(
                        violation['type'], violation['current_value'], violation['limit']
                    )
                    self.alert_system.send_alert(alert)
            
            # リスクレベルによる取引制限
            if risk_metrics.risk_level.value in ['high', 'extreme']:
                logger.warning(f"High risk level detected: {risk_metrics.risk_level.value}")
                # 新規ポジションを制限
                self.confidence_threshold = min(0.8, self.confidence_threshold + 0.1)
            else:
                # 通常設定に戻す
                self.confidence_threshold = max(0.4, self.confidence_threshold - 0.05)
            
            # 各銘柄の予測と取引判定
            for symbol in self.symbols:
                try:
                    if symbol not in current_prices:
                        continue
                    
                    current_price = current_prices[symbol]
                    
                    # AI予測実行
                    prediction = self.get_enhanced_prediction(symbol)
                    
                    # 予測履歴記録
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'price': current_price,
                        **prediction
                    }
                    self.prediction_history.append(prediction_record)
                    
                    # 現在ポジション確認
                    has_position = symbol in positions
                    
                    # リスク管理ベースのポジションサイズ計算
                    pos_recommendation = self.risk_manager.calculate_position_size(
                        symbol, account_summary['equity'], prediction['confidence'], current_price
                    )
                    
                    # 高信頼度シグナルのアラート送信
                    if prediction['confidence'] >= 0.8:
                        alert = self.alert_system.create_trading_signal_alert(
                            symbol, prediction['signal'], 
                            prediction['probability'], prediction['confidence'], current_price
                        )
                        self.alert_system.send_alert(alert)
                    
                    # 取引ロジック（リスク管理統合）
                    if prediction['signal'] == 'BUY' and not has_position and pos_recommendation.recommended_size > 0:
                        # リスク制限チェック
                        if len(risk_violations) == 0 or all(v['severity'] != 'high' for v in risk_violations):
                            order_id = self.trading_engine.place_order(
                                symbol, OrderSide.BUY, pos_recommendation.recommended_size, OrderType.MARKET
                            )
                            if order_id:
                                # ストップロス・テイクプロフィット設定
                                sl_tp = self.risk_manager.calculate_stop_loss_take_profit(
                                    symbol, current_price, 'long', prediction['confidence']
                                )
                                
                                # ストップロス注文（簡易実装）
                                stop_quantity = pos_recommendation.recommended_size
                                self.trading_engine.place_order(
                                    symbol, OrderSide.SELL, stop_quantity, OrderType.STOP_LOSS,
                                    stop_price=sl_tp['stop_loss_price']
                                )
                                
                                signal_record = {
                                    'timestamp': datetime.now().isoformat(),
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'quantity': pos_recommendation.recommended_size,
                                    'probability': prediction['probability'],
                                    'confidence': prediction['confidence'],
                                    'order_id': order_id,
                                    'stop_loss': sl_tp['stop_loss_price'],
                                    'take_profit': sl_tp['take_profit_price'],
                                    'risk_reason': pos_recommendation.reason
                                }
                                self.trade_signals.append(signal_record)
                                self.performance_stats['total_signals'] += 1
                                
                                logger.info(f"ENHANCED BUY: {symbol} {pos_recommendation.recommended_size} @ ${current_price:.2f} "
                                           f"(SL: ${sl_tp['stop_loss_price']:.2f})")
                    
                    elif prediction['signal'] == 'SELL' and has_position:
                        # ポジションクローズ
                        position = positions[symbol]
                        if position['side'] == 'long':
                            order_id = self.trading_engine.place_order(
                                symbol, OrderSide.SELL, position['quantity'], OrderType.MARKET
                            )
                            if order_id:
                                signal_record = {
                                    'timestamp': datetime.now().isoformat(),
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'price': current_price,
                                    'quantity': position['quantity'],
                                    'probability': prediction['probability'],
                                    'confidence': prediction['confidence'],
                                    'order_id': order_id,
                                    'entry_price': position['entry_price'],
                                    'pnl': position['unrealized_pnl']
                                }
                                self.trade_signals.append(signal_record)
                                
                                # パフォーマンス更新
                                if position['unrealized_pnl'] > 0:
                                    self.performance_stats['successful_trades'] += 1
                                    self._update_streak('win')
                                else:
                                    self.performance_stats['failed_trades'] += 1
                                    self._update_streak('loss')
                                
                                # パフォーマンス分析に記録
                                trade_record = {
                                    'symbol': symbol,
                                    'side': 'sell',
                                    'entry_time': datetime.now() - timedelta(hours=position.get('hold_time', 1)),
                                    'exit_time': datetime.now(),
                                    'timestamp': datetime.now(),
                                    'quantity': position['quantity'],
                                    'entry_price': position['entry_price'],
                                    'exit_price': current_price,
                                    'pnl': position['unrealized_pnl'],
                                    'fee': position['quantity'] * current_price * 0.001  # 0.1%手数料想定
                                }
                                self.performance_analyzer.add_trade(trade_record)
                                
                                logger.info(f"ENHANCED SELL: {symbol} {position['quantity']} @ ${current_price:.2f} "
                                           f"(PnL: ${position['unrealized_pnl']:.2f})")
                
                except Exception as e:
                    logger.error(f"Enhanced strategy error for {symbol}: {e}")
                    continue
            
            # リスクレポート生成
            risk_report = self.risk_manager.generate_risk_report(account_summary['equity'], positions)
            self.risk_reports.append(risk_report)
            
            # パフォーマンス分析に資産推移記録
            self.performance_analyzer.add_equity_point(datetime.now(), account_summary['equity'])
            
            # 予測結果をパフォーマンス分析に記録
            for pred in self.prediction_history[-len(self.symbols):]:  # 最新の予測
                self.performance_analyzer.add_prediction(pred)
            
            # 履歴サイズ制限
            for history_list in [self.prediction_history, self.trade_signals, self.risk_reports]:
                if len(history_list) > 500:
                    history_list[:] = history_list[-500:]
                    
        except Exception as e:
            logger.error(f"Enhanced strategy execution error: {e}")
    
    def _update_streak(self, result_type: str):
        """連勝・連敗記録更新"""
        if self.performance_stats['streak_type'] == result_type:
            self.performance_stats['current_streak'] += 1
        else:
            # Update max streak before resetting
            if self.performance_stats['streak_type'] == 'win':
                self.performance_stats['max_consecutive_wins'] = max(
                    self.performance_stats['max_consecutive_wins'],
                    self.performance_stats['current_streak']
                )
            elif self.performance_stats['streak_type'] == 'loss':
                self.performance_stats['max_consecutive_losses'] = max(
                    self.performance_stats['max_consecutive_losses'],
                    self.performance_stats['current_streak']
                )
            
            self.performance_stats['streak_type'] = result_type
            self.performance_stats['current_streak'] = 1
    
    def get_enhanced_summary(self) -> Dict:
        """強化サマリー取得"""
        account_summary = self.trading_engine.get_account_summary()
        positions = self.trading_engine.get_positions()
        
        # リスク指標
        risk_metrics = self.risk_manager.calculate_risk_metrics(account_summary['equity'], positions)
        risk_violations = self.risk_manager.check_risk_violations(account_summary['equity'], positions)
        
        # 最新予測
        latest_predictions = self.get_latest_predictions()
        
        return {
            'account': account_summary,
            'positions': positions,
            'risk_metrics': {
                'total_exposure': risk_metrics.total_exposure,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'var_95': risk_metrics.var_95,
                'portfolio_volatility': risk_metrics.portfolio_volatility,
                'concentration_risk': risk_metrics.concentration_risk,
                'risk_level': risk_metrics.risk_level.value
            },
            'risk_violations': risk_violations,
            'latest_predictions': latest_predictions,
            'performance_stats': self.performance_stats,
            'total_signals': len(self.trade_signals),
            'active_symbols': list(positions.keys())
        }
    
    def get_latest_predictions(self) -> List[Dict]:
        """最新予測結果取得"""
        latest_predictions = {}
        for pred in reversed(self.prediction_history):
            symbol = pred['symbol']
            if symbol not in latest_predictions:
                latest_predictions[symbol] = pred
        
        return list(latest_predictions.values())
    
    def get_prediction_engine_performance(self) -> Dict:
        """予測エンジン性能取得"""
        try:
            return self.prediction_engine.get_model_performance()
        except Exception as e:
            logger.error(f"予測エンジン性能取得エラー: {e}")
            return {}
    
    def force_model_retrain(self):
        """強制モデル再学習"""
        try:
            self.prediction_engine.force_retrain()
            logger.info("モデル再学習を強制実行しました")
        except Exception as e:
            logger.error(f"強制モデル再学習エラー: {e}")
    
    def get_advanced_summary(self) -> Dict:
        """高度サマリー取得（ML性能含む）"""
        # 基本サマリー取得
        summary = self.get_enhanced_summary()
        
        # ML性能情報追加
        ml_performance = self.get_prediction_engine_performance()
        summary['ml_performance'] = ml_performance
        
        # 予測精度統計
        if self.prediction_history:
            recent_predictions = self.prediction_history[-50:]  # 最新50予測
            high_confidence_count = sum(1 for p in recent_predictions if p.get('confidence', 0) > 0.7)
            avg_confidence = np.mean([p.get('confidence', 0) for p in recent_predictions])
            
            summary['prediction_stats'] = {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'high_confidence_predictions': high_confidence_count,
                'average_confidence': avg_confidence,
                'high_confidence_rate': high_confidence_count / len(recent_predictions) if recent_predictions else 0
            }
        
        return summary
    
    def get_alert_system_summary(self) -> Dict:
        """アラートシステムサマリー取得"""
        try:
            return {
                'alert_history': self.alert_system.get_alert_history(24),
                'alert_stats': self.alert_system.get_alert_stats(24),
                'channels_count': len(self.alert_system.channels),
                'enabled_channels': [ch.name for ch in self.alert_system.channels if ch.enabled]
            }
        except Exception as e:
            logger.error(f"アラートシステムサマリー取得エラー: {e}")
            return {}
    
    def update_alert_config(self, config: Dict):
        """アラート設定更新"""
        try:
            self.alert_system.update_config(config)
            logger.info("アラート設定を更新しました")
        except Exception as e:
            logger.error(f"アラート設定更新エラー: {e}")
    
    def send_test_alert(self):
        """テストアラート送信"""
        try:
            alert = self.alert_system.create_trading_signal_alert("TEST", "BUY", 0.75, 0.85, 50000)
            results = self.alert_system.send_alert(alert)
            logger.info(f"テストアラート送信結果: {results}")
            return results
        except Exception as e:
            logger.error(f"テストアラート送信エラー: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict:
        """パフォーマンスサマリー取得"""
        try:
            # 取引履歴をペーパートレーディングエンジンから取得
            trades = self.trading_engine.get_trade_history(limit=None)
            
            # パフォーマンス分析器の取引履歴と同期
            for trade in trades:
                if not any(t.get('id') == trade.get('id') for t in self.performance_analyzer.trade_history):
                    self.performance_analyzer.add_trade(trade)
            
            # メトリクス計算
            metrics = self.performance_analyzer.calculate_metrics(
                self.performance_analyzer.trade_history, 
                self.trading_engine.initial_balance
            )
            
            # 銘柄別パフォーマンス
            symbol_perfs = self.performance_analyzer.analyze_by_symbol(
                self.performance_analyzer.trade_history
            )
            
            # 予測精度分析
            prediction_analysis = self.performance_analyzer.analyze_predictions()
            
            return {
                'metrics': {
                    'total_return': metrics.total_return,
                    'total_return_pct': metrics.total_return_pct,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'total_trades': metrics.total_trades,
                    'avg_hold_time': metrics.avg_hold_time
                },
                'symbol_performance': [
                    {
                        'symbol': p.symbol,
                        'total_trades': p.total_trades,
                        'win_rate': p.win_rate,
                        'total_pnl': p.total_pnl,
                        'avg_pnl': p.avg_pnl
                    }
                    for p in symbol_perfs[:5]  # Top 5
                ],
                'prediction_accuracy': prediction_analysis,
                'equity_curve_points': len(self.performance_analyzer.equity_curve)
            }
        except Exception as e:
            logger.error(f"パフォーマンスサマリー取得エラー: {e}")
            return {}
    
    def generate_performance_report(self, format: str = 'html') -> str:
        """パフォーマンスレポート生成"""
        try:
            if format == 'html':
                filepath = self.performance_analyzer.generate_html_report()
            elif format == 'pdf':
                filepath = self.performance_analyzer.generate_pdf_report()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"パフォーマンスレポート生成完了: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return ""
    
    def export_trade_history(self, format: str = 'csv') -> str:
        """取引履歴エクスポート"""
        try:
            filepath = self.performance_analyzer.export_trade_history(format=format)
            logger.info(f"取引履歴エクスポート完了: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"取引履歴エクスポートエラー: {e}")
            return ""
    
    # === マルチ銘柄対応メソッド ===
    
    def get_multi_symbol_summary(self) -> Dict:
        """マルチ銘柄サマリー取得"""
        try:
            # マルチ銘柄管理システムから取引サマリーを取得
            trading_summary = self.multi_symbol_manager.generate_trading_summary()
            
            # 各銘柄の予測データを更新
            current_prices = self.get_live_prices()
            for symbol in self.multi_symbol_manager.get_enabled_symbols():
                if symbol in current_prices:
                    prediction = self.get_enhanced_prediction(symbol)
                    self.multi_symbol_manager.update_live_data(symbol, current_prices[symbol], prediction)
            
            # 取引機会の分析
            opportunities = self.multi_symbol_manager.get_trading_opportunities()
            
            # ポートフォリオエクスポージャー
            positions = self.trading_engine.get_positions()
            position_values = {symbol: pos.get('market_value', 0) for symbol, pos in positions.items()}
            exposure_analysis = self.multi_symbol_manager.get_portfolio_exposure(position_values)
            
            # 銘柄別パフォーマンス指標
            performance_metrics = self.multi_symbol_manager.get_symbol_performance_metrics()
            
            return {
                'trading_summary': trading_summary,
                'trading_opportunities': opportunities,
                'exposure_analysis': exposure_analysis,
                'symbol_metrics': performance_metrics,
                'correlation_matrix': self.multi_symbol_manager.correlation_matrix.to_dict() if not self.multi_symbol_manager.correlation_matrix.empty else {}
            }
            
        except Exception as e:
            logger.error(f"マルチ銘柄サマリー取得エラー: {e}")
            return {}
    
    def enable_symbol_trading(self, symbol: str):
        """銘柄取引を有効化"""
        try:
            self.multi_symbol_manager.enable_symbol(symbol)
            self.symbols = self.multi_symbol_manager.get_enabled_symbols()
            logger.info(f"{symbol} 取引を有効化。現在の対象銘柄: {self.symbols}")
        except Exception as e:
            logger.error(f"{symbol} 有効化エラー: {e}")
    
    def disable_symbol_trading(self, symbol: str):
        """銘柄取引を無効化"""
        try:
            self.multi_symbol_manager.disable_symbol(symbol)
            self.symbols = self.multi_symbol_manager.get_enabled_symbols()
            logger.info(f"{symbol} 取引を無効化。現在の対象銘柄: {self.symbols}")
        except Exception as e:
            logger.error(f"{symbol} 無効化エラー: {e}")
    
    def update_symbol_config(self, symbol: str, **kwargs):
        """銘柄設定を更新"""
        try:
            self.multi_symbol_manager.update_symbol_config(symbol, **kwargs)
            logger.info(f"{symbol} 設定を更新: {kwargs}")
        except Exception as e:
            logger.error(f"{symbol} 設定更新エラー: {e}")
    
    def get_symbol_config(self, symbol: str) -> Optional[Dict]:
        """銘柄設定を取得"""
        try:
            config = self.multi_symbol_manager.get_symbol_config(symbol)
            return config.to_dict() if config else None
        except Exception as e:
            logger.error(f"{symbol} 設定取得エラー: {e}")
            return None
    
    def execute_multi_symbol_strategy(self):
        """マルチ銘柄取引戦略実行"""
        try:
            logger.info("マルチ銘柄取引戦略を実行中...")
            
            # 各有効銘柄に対して取引戦略実行
            current_prices = self.get_live_prices()
            all_predictions = {}
            
            for symbol in self.multi_symbol_manager.get_enabled_symbols():
                if symbol in current_prices:
                    # 予測取得
                    prediction = self.get_enhanced_prediction(symbol)
                    all_predictions[symbol] = prediction
                    
                    # ライブデータ更新
                    self.multi_symbol_manager.update_live_data(symbol, current_prices[symbol], prediction)
                    
                    # 銘柄設定取得
                    config = self.multi_symbol_manager.get_symbol_config(symbol)
                    if not config:
                        continue
                    
                    # 個別銘柄取引ロジック実行
                    self._execute_symbol_strategy(symbol, prediction, config, current_prices[symbol])
            
            # 取引機会の分析と実行
            opportunities = self.multi_symbol_manager.get_trading_opportunities()
            if opportunities:
                logger.info(f"{len(opportunities)}件の取引機会を検出")
                
                # 最高信頼度の機会から処理
                for opportunity in opportunities[:3]:  # 上位3件
                    self._process_trading_opportunity(opportunity)
            
            # 相関行列更新
            self.multi_symbol_manager.calculate_correlation_matrix()
            
            logger.info("マルチ銘柄取引戦略実行完了")
            
        except Exception as e:
            logger.error(f"マルチ銘柄戦略実行エラー: {e}")
    
    def _execute_symbol_strategy(self, symbol: str, prediction: Dict, config, current_price: float):
        """個別銘柄戦略実行"""
        try:
            confidence = prediction.get('confidence', 0)
            signal = prediction.get('signal', 'HOLD')
            
            # 最小信頼度チェック
            if confidence < config.min_confidence:
                return
            
            # リスクチェック
            if not self.risk_manager.can_open_position(symbol, config.max_position_size):
                return
            
            # ポジション管理
            current_positions = self.trading_engine.get_positions()
            
            if signal == 'BUY' and symbol not in current_positions:
                # 新規買いポジション
                position_size = min(config.max_position_size, 
                                  self.risk_manager.calculate_optimal_position_size(
                                      symbol, current_price, confidence))
                
                if position_size > 0:
                    self.trading_engine.place_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=position_size,
                        price=current_price
                    )
                    logger.info(f"{symbol} 買いポジション開始: {position_size}")
            
            elif signal == 'SELL' and symbol in current_positions:
                # ポジション決済
                position = current_positions[symbol]
                if position['quantity'] > 0:
                    self.trading_engine.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=position['quantity'],
                        price=current_price
                    )
                    logger.info(f"{symbol} ポジション決済: {position['quantity']}")
                    
        except Exception as e:
            logger.error(f"{symbol} 個別戦略実行エラー: {e}")
    
    def _process_trading_opportunity(self, opportunity: Dict):
        """取引機会処理"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            confidence = opportunity['confidence']
            
            # アラート送信
            if confidence >= 0.85:  # 高信頼度の場合
                alert = self.alert_system.create_trading_signal_alert(
                    symbol, signal, confidence, opportunity.get('price', 0)
                )
                self.alert_system.send_alert(alert)
                
        except Exception as e:
            logger.error(f"取引機会処理エラー: {e}")
    
    def save_multi_symbol_config(self):
        """マルチ銘柄設定を保存"""
        try:
            self.multi_symbol_manager.save_config()
            logger.info("マルチ銘柄設定を保存しました")
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")

# テスト関数
def test_enhanced_ai_trader():
    """強化AIトレーダーのテスト"""
    print("=== Enhanced AI Trader Test ===")
    
    trader = EnhancedAITrader(10000.0)
    
    # 初期状態
    summary = trader.get_enhanced_summary()
    print(f"Initial Balance: ${summary['account']['balance']:,.2f}")
    print(f"Risk Level: {summary['risk_metrics']['risk_level']}")
    
    # 強化取引戦略実行
    print("\nExecuting enhanced trading strategy...")
    trader.execute_enhanced_strategy()
    
    # 結果確認
    summary = trader.get_enhanced_summary()
    print(f"\nEnhanced Results:")
    print(f"Equity: ${summary['account']['equity']:,.2f}")
    print(f"Risk Level: {summary['risk_metrics']['risk_level']}")
    print(f"Total Exposure: {summary['risk_metrics']['total_exposure']:.1%}")
    print(f"Portfolio Volatility: {summary['risk_metrics']['portfolio_volatility']:.1%}")
    print(f"Total Signals: {summary['total_signals']}")
    print(f"Active Positions: {len(summary['positions'])}")
    
    # 予測詳細
    predictions = summary['latest_predictions']
    if predictions:
        print(f"\nLatest Enhanced Predictions:")
        for pred in predictions:
            print(f"  {pred['symbol']}: {pred['probability']:.1%} ({pred['signal']}) "
                  f"Confidence: {pred['confidence']:.1%}")
    
    # リスク警告
    violations = summary['risk_violations']
    if violations:
        print(f"\nRisk Violations ({len(violations)}):")
        for violation in violations[:3]:
            print(f"  {violation['severity'].upper()}: {violation['message']}")

if __name__ == "__main__":
    test_enhanced_ai_trader()