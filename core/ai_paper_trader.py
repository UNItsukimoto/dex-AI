#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI ペーパートレーダー
AI予測とペーパートレーディングエンジンの統合システム
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIPaperTrader:
    """AI駆動ペーパートレーダー"""
    
    def __init__(self, initial_balance: float = 10000.0):
        # ペーパートレーディングエンジン初期化
        self.trading_engine = PaperTradingEngine(initial_balance)
        self.trading_engine.max_position_size = 0.2  # 最大ポジション20%に調整
        
        # API設定
        self.api_base_url = "https://api.hyperliquid.xyz"
        
        # 取引設定
        self.symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
        self.position_size_pct = 0.05  # 1回の取引で資金の5%
        self.confidence_threshold = 0.6  # 予測信頼度閾値
        self.buy_threshold = 0.65   # 買いシグナル閾値
        self.sell_threshold = 0.35  # 売りシグナル閾値
        
        # データ保存
        self.price_history: Dict[str, List] = {symbol: [] for symbol in self.symbols}
        self.prediction_history: List[Dict] = []
        self.trade_signals: List[Dict] = []
        
        logger.info(f"AI Paper Trader initialized with ${initial_balance:,.2f}")
    
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
    
    def get_candles_for_prediction(self, symbol: str, count: int = 50) -> pd.DataFrame:
        """予測用ローソク足データ取得"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (count * 60 * 60 * 1000)  # 1時間足
            
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
    
    def create_prediction_features(self, df: pd.DataFrame) -> Dict:
        """予測特徴量作成（改良版）"""
        if len(df) < 20:
            return {
                'probability': 0.5,
                'confidence': 0.0,
                'signal': 'HOLD',
                'features': {}
            }
        
        try:
            # 基本特徴量
            df['price_change_1h'] = df['close'].pct_change()
            df['price_change_4h'] = df['close'].pct_change(4)
            df['price_change_12h'] = df['close'].pct_change(12)
            
            # 移動平均
            df['ma_7'] = df['close'].rolling(7).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_ratio'] = df['close'] / df['ma_20']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ボラティリティ
            df['volatility'] = df['close'].rolling(10).std() / df['close']
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # 最新データで予測
            latest = df.iloc[-1]
            
            # 改良された予測アルゴリズム
            prediction_score = 0.0
            confidence_factors = []
            
            # 1. トレンド分析 (30%)
            trend_score = 0
            if latest['ma_ratio'] > 1.02:  # 2%以上上昇
                trend_score += 0.15
            elif latest['ma_ratio'] > 1.0:
                trend_score += 0.05
            elif latest['ma_ratio'] < 0.98:  # 2%以上下落
                trend_score -= 0.15
            elif latest['ma_ratio'] < 1.0:
                trend_score -= 0.05
            
            if latest['ma_7'] > latest['ma_20']:
                trend_score += 0.10
            else:
                trend_score -= 0.10
            
            prediction_score += trend_score
            confidence_factors.append(abs(trend_score))
            
            # 2. モメンタム分析 (25%)
            momentum_score = 0
            if latest['price_change_1h'] > 0.01:  # 1%以上上昇
                momentum_score += 0.10
            elif latest['price_change_1h'] > 0:
                momentum_score += 0.05
            elif latest['price_change_1h'] < -0.01:  # 1%以上下落
                momentum_score -= 0.10
            else:
                momentum_score -= 0.05
            
            if latest['price_change_4h'] > 0:
                momentum_score += 0.08
            else:
                momentum_score -= 0.08
            
            if latest['price_change_12h'] > 0:
                momentum_score += 0.07
            else:
                momentum_score -= 0.07
            
            prediction_score += momentum_score
            confidence_factors.append(abs(momentum_score))
            
            # 3. RSI分析 (20%)
            rsi_score = 0
            if latest['rsi'] < 30:  # 売られすぎ
                rsi_score += 0.15
            elif latest['rsi'] < 40:
                rsi_score += 0.08
            elif latest['rsi'] > 70:  # 買われすぎ
                rsi_score -= 0.15
            elif latest['rsi'] > 60:
                rsi_score -= 0.08
            
            prediction_score += rsi_score
            confidence_factors.append(abs(rsi_score))
            
            # 4. MACD分析 (15%)
            macd_score = 0
            if latest['macd'] > latest['macd_signal']:
                macd_score += 0.08
            else:
                macd_score -= 0.08
            
            # MACD傾斜
            if len(df) > 2:
                macd_slope = latest['macd'] - df['macd'].iloc[-2]
                if macd_slope > 0:
                    macd_score += 0.07
                else:
                    macd_score -= 0.07
            
            prediction_score += macd_score
            confidence_factors.append(abs(macd_score))
            
            # 5. ボラティリティ調整 (10%)
            vol_adjustment = 0
            if latest['volatility'] > df['volatility'].rolling(20).mean().iloc[-1]:
                # 高ボラティリティ時は予測を控えめに
                prediction_score *= 0.9
                vol_adjustment = -0.05
            else:
                vol_adjustment = 0.05
            
            confidence_factors.append(abs(vol_adjustment))
            
            # 最終確率計算
            probability = 0.5 + prediction_score
            probability = max(0.1, min(0.9, probability))
            
            # 信頼度計算
            confidence = np.mean(confidence_factors) * min(1.0, len(df) / 30.0)
            confidence = max(0.1, min(0.9, confidence))
            
            # シグナル判定
            if probability >= self.buy_threshold and confidence >= self.confidence_threshold:
                signal = 'BUY'
            elif probability <= self.sell_threshold and confidence >= self.confidence_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            features = {
                'price': latest['close'],
                'ma_ratio': latest['ma_ratio'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'volatility': latest['volatility'],
                'price_change_1h': latest['price_change_1h'],
                'price_change_4h': latest['price_change_4h'],
                'price_change_12h': latest['price_change_12h']
            }
            
            return {
                'probability': probability,
                'confidence': confidence,
                'signal': signal,
                'features': features,
                'prediction_score': prediction_score
            }
            
        except Exception as e:
            logger.error(f"Prediction features error: {e}")
            return {
                'probability': 0.5,
                'confidence': 0.0,
                'signal': 'HOLD',
                'features': {}
            }
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """ポジションサイズ計算"""
        account_summary = self.trading_engine.get_account_summary()
        available_capital = account_summary['equity'] * self.position_size_pct
        
        # 最小取引単位を考慮
        min_notional = 10.0  # 最小$10の取引
        max_quantity = available_capital / price
        
        if available_capital < min_notional:
            return 0.0
        
        # 銘柄別の最小取引単位
        if symbol == 'BTC':
            min_quantity = 0.001
        elif symbol == 'ETH':
            min_quantity = 0.01
        elif symbol in ['SOL', 'AVAX']:
            min_quantity = 0.1
        else:
            min_quantity = 0.001
        
        quantity = max(min_quantity, max_quantity)
        
        # 小数点調整
        if symbol == 'BTC':
            quantity = round(quantity, 3)
        elif symbol == 'ETH':
            quantity = round(quantity, 2)
        else:
            quantity = round(quantity, 1)
        
        return quantity
    
    def execute_trading_strategy(self):
        """取引戦略実行"""
        try:
            logger.info("Executing AI trading strategy...")
            
            # 現在価格取得
            current_prices = self.get_live_prices()
            if not current_prices:
                logger.warning("No price data available")
                return
            
            # 価格更新
            self.trading_engine.update_prices(current_prices)
            
            # 各銘柄の予測と取引判定
            for symbol in self.symbols:
                try:
                    if symbol not in current_prices:
                        continue
                    
                    current_price = current_prices[symbol]
                    
                    # 予測データ取得
                    df = self.get_candles_for_prediction(symbol)
                    if df.empty:
                        logger.warning(f"No candle data for {symbol}")
                        continue
                    
                    # AI予測実行
                    prediction = self.create_prediction_features(df)
                    
                    # 予測履歴記録
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'price': current_price,
                        **prediction
                    }
                    self.prediction_history.append(prediction_record)
                    
                    # 現在ポジション確認
                    positions = self.trading_engine.get_positions()
                    has_position = symbol in positions
                    
                    # 取引ロジック
                    if prediction['signal'] == 'BUY' and not has_position:
                        # 新規買いポジション
                        quantity = self.calculate_position_size(symbol, current_price)
                        if quantity > 0:
                            order_id = self.trading_engine.place_order(
                                symbol, OrderSide.BUY, quantity, OrderType.MARKET
                            )
                            if order_id:
                                signal_record = {
                                    'timestamp': datetime.now().isoformat(),
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'quantity': quantity,
                                    'probability': prediction['probability'],
                                    'confidence': prediction['confidence'],
                                    'order_id': order_id
                                }
                                self.trade_signals.append(signal_record)
                                logger.info(f"BUY signal executed: {symbol} {quantity} @ ${current_price:.2f}")
                    
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
                                    'order_id': order_id
                                }
                                self.trade_signals.append(signal_record)
                                logger.info(f"SELL signal executed: {symbol} {position['quantity']} @ ${current_price:.2f}")
                    
                    # 価格履歴記録
                    self.price_history[symbol].append({
                        'timestamp': datetime.now(),
                        'price': current_price
                    })
                    
                    # 履歴サイズ制限
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]
                    
                except Exception as e:
                    logger.error(f"Trading strategy error for {symbol}: {e}")
                    continue
            
            # 予測履歴サイズ制限
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # 取引シグナル履歴制限
            if len(self.trade_signals) > 500:
                self.trade_signals = self.trade_signals[-500:]
                
        except Exception as e:
            logger.error(f"Trading strategy execution error: {e}")
    
    def get_trading_summary(self) -> Dict:
        """取引サマリー取得"""
        account_summary = self.trading_engine.get_account_summary()
        positions = self.trading_engine.get_positions()
        recent_trades = self.trading_engine.get_trade_history(10)
        
        return {
            'account': account_summary,
            'positions': positions,
            'recent_trades': recent_trades,
            'prediction_count': len(self.prediction_history),
            'signal_count': len(self.trade_signals),
            'active_symbols': list(positions.keys())
        }
    
    def get_latest_predictions(self) -> List[Dict]:
        """最新予測結果取得"""
        # 各銘柄の最新予測を取得
        latest_predictions = {}
        for pred in reversed(self.prediction_history):
            symbol = pred['symbol']
            if symbol not in latest_predictions:
                latest_predictions[symbol] = pred
        
        return list(latest_predictions.values())
    
    def save_trading_state(self):
        """取引状態保存"""
        self.trading_engine.save_state()
        logger.info("AI trading state saved")

# テスト関数
def test_ai_paper_trader():
    """AI ペーパートレーダーのテスト"""
    print("=== AI Paper Trader Test ===")
    
    trader = AIPaperTrader(10000.0)
    
    # 初期状態
    summary = trader.get_trading_summary()
    print(f"Initial Balance: ${summary['account']['balance']:,.2f}")
    
    # 取引戦略実行
    print("\nExecuting trading strategy...")
    trader.execute_trading_strategy()
    
    # 結果確認
    summary = trader.get_trading_summary()
    print(f"\nResults:")
    print(f"Equity: ${summary['account']['equity']:,.2f}")
    print(f"Total Trades: {summary['account']['total_trades']}")
    print(f"Active Positions: {len(summary['positions'])}")
    
    # 最新予測
    predictions = trader.get_latest_predictions()
    print(f"\nLatest Predictions ({len(predictions)}):")
    for pred in predictions:
        print(f"  {pred['symbol']}: {pred['probability']:.1%} ({pred['signal']}) "
              f"Confidence: {pred['confidence']:.1%}")
    
    # ポジション詳細
    if summary['positions']:
        print(f"\nActive Positions:")
        for symbol, pos in summary['positions'].items():
            print(f"  {symbol}: {pos['side']} {pos['quantity']} @ ${pos['entry_price']:.2f} "
                  f"PnL: ${pos['unrealized_pnl']:.2f}")

if __name__ == "__main__":
    test_ai_paper_trader()