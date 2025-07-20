#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルな体験のペーパートレーディング
実際の取引により近い体験を提供
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
import random
from dataclasses import dataclass
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class RealisticOrder:
    """リアルな注文オブジェクト"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0

@dataclass
class RealisticPosition:
    """リアルなポジション"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100

class RealisticPaperTradingEngine:
    """リアルな体験のペーパートレーディングエンジン"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        # 注文とポジション
        self.orders: Dict[str, RealisticOrder] = {}
        self.positions: Dict[str, RealisticPosition] = {}
        self.trade_history: List[Dict] = []
        
        # リアルな市場データ
        self.live_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        # 取引設定（リアルな体験のため）
        self.transaction_fee_rate = 0.001  # 0.1%の手数料
        self.slippage_rate = 0.0005  # 0.05%のスリッページ
        self.order_fill_delay = (0.1, 1.0)  # 0.1-1秒の約定遅延
        
        # サポート銘柄
        self.supported_symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE', 'MATIC']
        
        logger.info(f"Realistic Paper Trading Engine initialized with ${initial_balance:,.2f}")
        
        # 初期価格設定
        self._initialize_prices()
        
        # バックグラウンドで価格更新開始
        self._start_price_updates()
    
    def _initialize_prices(self):
        """初期価格設定"""
        base_prices = {
            'BTC': 45000,
            'ETH': 3200,
            'SOL': 150,
            'AVAX': 35,
            'DOGE': 0.08,
            'MATIC': 0.75
        }
        
        for symbol, price in base_prices.items():
            # 実際のAPIから価格を取得を試行、失敗時はベース価格使用
            real_price = self._fetch_real_price(symbol)
            self.live_prices[symbol] = real_price if real_price else price
            self.price_history[symbol] = []
    
    def _fetch_real_price(self, symbol: str) -> Optional[float]:
        """実際の価格を取得（可能な場合）"""
        try:
            # Hyperliquid APIから実際の価格を取得
            response = requests.get("https://api.hyperliquid.xyz/info", 
                                  json={"type": "allMids"}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if symbol in data:
                    return float(data[symbol])
            
        except Exception as e:
            logger.debug(f"実際の価格取得失敗 {symbol}: {e}")
        
        return None
    
    def _start_price_updates(self):
        """価格更新の開始（シミュレート）"""
        # 実際の実装では別スレッドで動作
        pass
    
    def update_live_prices(self):
        """ライブ価格の更新"""
        for symbol in self.supported_symbols:
            if symbol in self.live_prices:
                # 実際の価格を取得を試行
                real_price = self._fetch_real_price(symbol)
                
                if real_price:
                    self.live_prices[symbol] = real_price
                else:
                    # リアルな価格変動をシミュレート
                    current_price = self.live_prices[symbol]
                    volatility = self._get_symbol_volatility(symbol)
                    change = np.random.normal(0, volatility) * current_price
                    new_price = max(current_price + change, current_price * 0.8)  # 20%以下には下がらない
                    self.live_prices[symbol] = new_price
                
                # 価格履歴に追加
                self.price_history[symbol].append({
                    'price': self.live_prices[symbol],
                    'timestamp': datetime.now()
                })
                
                # 履歴は最新1000件まで保持
                if len(self.price_history[symbol]) > 1000:
                    self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # ポジションの未実現損益を更新
        self._update_unrealized_pnl()
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """銘柄のボラティリティ"""
        volatilities = {
            'BTC': 0.02,    # 2%
            'ETH': 0.025,   # 2.5%
            'SOL': 0.04,    # 4%
            'AVAX': 0.05,   # 5%
            'DOGE': 0.08,   # 8%
            'MATIC': 0.06   # 6%
        }
        return volatilities.get(symbol, 0.03)
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None) -> Optional[str]:
        """注文実行（リアルな体験）"""
        try:
            # 基本バリデーション
            if symbol not in self.supported_symbols:
                logger.error(f"サポートされていない銘柄: {symbol}")
                return None
            
            if quantity <= 0:
                logger.error("数量は0より大きくしてください")
                return None
            
            # 現在価格取得
            current_price = self.live_prices.get(symbol)
            if not current_price:
                logger.error(f"{symbol}の価格データがありません")
                return None
            
            # 市場価格注文の場合の価格設定
            if order_type == OrderType.MARKET:
                price = current_price
            
            # 残高チェック（買い注文の場合）
            if side == OrderSide.BUY:
                required_amount = quantity * price
                if required_amount > self.balance:
                    logger.error(f"残高不足: 必要 ${required_amount:,.2f}, 利用可能 ${self.balance:,.2f}")
                    return None
            
            # 売りポジションチェック（売り注文の場合）
            if side == OrderSide.SELL:
                current_position = self.positions.get(symbol)
                if not current_position or current_position.quantity < quantity:
                    available = current_position.quantity if current_position else 0
                    logger.error(f"ポジション不足: 必要 {quantity}, 利用可能 {available}")
                    return None
            
            # 注文作成
            order_id = str(uuid.uuid4())[:8]
            order = RealisticOrder(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            logger.info(f"注文受付: {order_id} {symbol} {side.value} {quantity}")
            
            # 市場価格注文は即座に約定
            if order_type == OrderType.MARKET:
                self._fill_order(order_id)
            
            return order_id
            
        except Exception as e:
            logger.error(f"注文エラー: {e}")
            return None
    
    def _fill_order(self, order_id: str):
        """注文約定処理"""
        try:
            order = self.orders.get(order_id)
            if not order or order.status != OrderStatus.PENDING:
                return
            
            # 約定遅延をシミュレート
            delay = random.uniform(*self.order_fill_delay)
            time.sleep(delay)
            
            # 現在価格と約定価格の計算
            current_price = self.live_prices[order.symbol]
            
            # スリッページを考慮した約定価格
            if order.side == OrderSide.BUY:
                slippage = random.uniform(0, self.slippage_rate)
                fill_price = current_price * (1 + slippage)
            else:
                slippage = random.uniform(0, self.slippage_rate)
                fill_price = current_price * (1 - slippage)
            
            # 手数料計算
            trade_value = order.quantity * fill_price
            fee = trade_value * self.transaction_fee_rate
            
            # 約定情報更新
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.fee = fee
            order.slippage = abs(fill_price - current_price) / current_price
            
            # ポジション更新
            self._update_position(order)
            
            # 残高更新
            if order.side == OrderSide.BUY:
                self.balance -= (trade_value + fee)
            else:
                self.balance += (trade_value - fee)
            
            # 取引履歴に追加
            self.trade_history.append({
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': fill_price,
                'fee': fee,
                'slippage': order.slippage,
                'timestamp': order.filled_at.isoformat(),
                'status': 'filled'
            })
            
            logger.info(f"約定完了: {order.id} {order.symbol} {order.quantity} @ ${fill_price:,.2f}")
            
        except Exception as e:
            logger.error(f"約定処理エラー: {e}")
    
    def _update_position(self, order: RealisticOrder):
        """ポジション更新"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # 新規ポジション
            if order.side == OrderSide.BUY:
                self.positions[symbol] = RealisticPosition(
                    symbol=symbol,
                    quantity=order.quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=order.filled_at
                )
        else:
            # 既存ポジション更新
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # ポジション増加
                total_quantity = position.quantity + order.quantity
                total_cost = (position.quantity * position.entry_price + 
                            order.quantity * order.filled_price)
                
                position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
                
            else:
                # ポジション減少
                position.quantity -= order.quantity
                
                # ポジションがゼロ以下になった場合
                if position.quantity <= 0:
                    del self.positions[symbol]
    
    def _update_unrealized_pnl(self):
        """未実現損益の更新"""
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            current_price = self.live_prices.get(symbol, position.entry_price)
            position.current_price = current_price
            
            # 未実現損益計算
            price_diff = current_price - position.entry_price
            position.unrealized_pnl = price_diff * position.quantity
            total_unrealized_pnl += position.unrealized_pnl
        
        # エクイティ更新
        self.equity = self.balance + total_unrealized_pnl
    
    def get_account_summary(self) -> Dict:
        """アカウントサマリー"""
        self.update_live_prices()
        
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': total_unrealized_pnl,
            'total_position_value': total_position_value,
            'free_margin': self.balance,
            'margin_ratio': (total_position_value / self.equity) if self.equity > 0 else 0,
            'daily_pnl': self.equity - self.initial_balance,
            'daily_pnl_pct': ((self.equity - self.initial_balance) / self.initial_balance) if self.initial_balance > 0 else 0
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """ポジション一覧"""
        self.update_live_prices()
        
        positions_dict = {}
        for symbol, position in self.positions.items():
            positions_dict[symbol] = {
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.pnl_percentage,
                'entry_time': position.entry_time.isoformat()
            }
        
        return positions_dict
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """取引履歴"""
        return self.trade_history[-limit:] if limit else self.trade_history
    
    def get_live_prices(self) -> Dict[str, float]:
        """現在価格"""
        self.update_live_prices()
        return self.live_prices.copy()
    
    def get_price_history(self, symbol: str, period_hours: int = 24) -> List[Dict]:
        """価格履歴"""
        if symbol not in self.price_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        return [
            entry for entry in self.price_history[symbol]
            if entry['timestamp'] >= cutoff_time
        ]
    
    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.info(f"注文キャンセル: {order_id}")
            return True
        
        return False
    
    def get_market_summary(self) -> Dict:
        """マーケットサマリー"""
        self.update_live_prices()
        
        summary = {}
        for symbol in self.supported_symbols:
            price = self.live_prices.get(symbol, 0)
            history = self.get_price_history(symbol, 24)
            
            if history:
                prices = [entry['price'] for entry in history]
                day_open = prices[0] if prices else price
                day_high = max(prices) if prices else price
                day_low = min(prices) if prices else price
                change_24h = ((price - day_open) / day_open * 100) if day_open > 0 else 0
            else:
                day_open = day_high = day_low = price
                change_24h = 0
            
            summary[symbol] = {
                'price': price,
                'change_24h': change_24h,
                'day_high': day_high,
                'day_low': day_low,
                'day_open': day_open
            }
        
        return summary