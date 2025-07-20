#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ペーパートレーディングエンジン
リスクゼロでのAI予測システム実践テスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """注文サイド"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """ポジションサイド"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """注文クラス"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Position:
    """ポジションクラス"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def update_price(self, new_price: float):
        """価格更新とPnL計算"""
        self.current_price = new_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity

@dataclass
class Account:
    """アカウント情報"""
    balance: float = 10000.0  # 初期資金 $10,000
    equity: float = 10000.0
    margin_used: float = 0.0
    margin_free: float = 10000.0
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

class PaperTradingEngine:
    """ペーパートレーディングエンジン"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance  # 後方互換性のため追加
        self.account = Account(balance=initial_balance, equity=initial_balance, margin_free=initial_balance)
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        self.price_data: Dict[str, float] = {}
        
        # 取引設定
        self.max_position_size = 0.1  # 最大ポジションサイズ（資金の10%）
        self.transaction_fee = 0.001  # 取引手数料（0.1%）
        self.slippage = 0.0005  # スリッページ（0.05%）
        
        # 取引ログ保存先
        self.log_file = Path("data/paper_trading_log.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
        logger.info(f"Paper Trading Engine initialized with ${initial_balance:,.2f}")
    
    def update_prices(self, prices: Dict[str, float]):
        """価格データ更新"""
        self.price_data.update(prices)
        
        # ポジションのPnL更新
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
        
        # アカウント情報更新
        self._update_account_equity()
        
        # ペンディング注文のチェック
        self._check_pending_orders()
    
    def _update_account_equity(self):
        """アカウント資産更新"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.account.equity = self.account.balance + unrealized_pnl
        self.account.margin_free = self.account.equity - self.account.margin_used
    
    def _check_pending_orders(self):
        """ペンディング注文の執行チェック"""
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING:
                if self._should_fill_order(order):
                    self._fill_order(order)
    
    def _should_fill_order(self, order: Order) -> bool:
        """注文執行判定"""
        if order.symbol not in self.price_data:
            return False
        
        current_price = self.price_data[order.symbol]
        
        if order.type == OrderType.MARKET:
            return True
        elif order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:
                return current_price >= order.price
        elif order.type == OrderType.STOP_LOSS:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price
        
        return False
    
    def _fill_order(self, order: Order):
        """注文執行"""
        try:
            current_price = self.price_data[order.symbol]
            
            # 執行価格計算（スリッページ考慮）
            if order.type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    fill_price = current_price * (1 + self.slippage)
                else:
                    fill_price = current_price * (1 - self.slippage)
            else:
                fill_price = order.price
            
            # 手数料計算
            notional = order.quantity * fill_price
            fee = notional * self.transaction_fee
            
            # 資金チェック
            if order.side == OrderSide.BUY:
                required_margin = notional + fee
                if required_margin > self.account.margin_free:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Order {order.id} rejected: insufficient funds")
                    return
            
            # 注文執行
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            
            # ポジション更新
            self._update_position(order)
            
            # アカウント更新
            if order.side == OrderSide.BUY:
                self.account.balance -= (notional + fee)
                self.account.margin_used += notional
            else:
                self.account.balance += (notional - fee)
                self.account.margin_used -= notional
            
            # 取引履歴記録
            trade = {
                'id': order.id,
                'timestamp': order.filled_at.isoformat(),
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': fill_price,
                'fee': fee,
                'pnl': 0.0  # ポジション閉鎖時に計算
            }
            self.trade_history.append(trade)
            self.account.total_trades += 1
            
            logger.info(f"Order filled: {order.symbol} {order.side.value} {order.quantity} @ ${fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error filling order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
    
    def _update_position(self, order: Order):
        """ポジション更新"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # 新規ポジション
            side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=order.quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price
            )
        else:
            # 既存ポジション更新
            position = self.positions[symbol]
            
            if (position.side == PositionSide.LONG and order.side == OrderSide.BUY) or \
               (position.side == PositionSide.SHORT and order.side == OrderSide.SELL):
                # ポジション増加
                total_cost = (position.quantity * position.entry_price + 
                             order.quantity * order.filled_price)
                total_quantity = position.quantity + order.quantity
                position.entry_price = total_cost / total_quantity
                position.quantity = total_quantity
            else:
                # ポジション減少またはクローズ
                if order.quantity >= position.quantity:
                    # ポジションクローズ
                    pnl = self._calculate_position_pnl(position, order.filled_price)
                    position.realized_pnl += pnl
                    self.account.total_pnl += pnl
                    
                    if pnl > 0:
                        self.account.winning_trades += 1
                    else:
                        self.account.losing_trades += 1
                    
                    # 取引履歴のPnL更新
                    if self.trade_history:
                        self.trade_history[-1]['pnl'] = pnl
                    
                    del self.positions[symbol]
                    logger.info(f"Position closed: {symbol} PnL: ${pnl:.2f}")
                else:
                    # 部分クローズ
                    position.quantity -= order.quantity
    
    def _calculate_position_pnl(self, position: Position, close_price: float) -> float:
        """ポジションPnL計算"""
        if position.side == PositionSide.LONG:
            return (close_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - close_price) * position.quantity
    
    def place_order(self, symbol: str, side: OrderSide, quantity: float, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """注文発注"""
        try:
            # ポジションサイズチェック
            if symbol in self.price_data:
                notional = quantity * self.price_data[symbol]
                max_notional = self.account.equity * self.max_position_size
                
                if notional > max_notional:
                    logger.warning(f"Order size too large: ${notional:.2f} > ${max_notional:.2f}")
                    return None
            
            order_id = str(uuid.uuid4())[:8]
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            self.orders[order_id] = order
            
            # マーケット注文は即座に執行
            if order_type == OrderType.MARKET and symbol in self.price_data:
                self._fill_order(order)
            
            logger.info(f"Order placed: {order_id} {symbol} {side.value} {quantity}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELED
                logger.info(f"Order canceled: {order_id}")
                return True
        return False
    
    def get_account_summary(self) -> Dict:
        """アカウントサマリー取得"""
        win_rate = (self.account.winning_trades / max(1, self.account.total_trades)) * 100
        
        return {
            'balance': self.account.balance,
            'equity': self.account.equity,
            'margin_used': self.account.margin_used,
            'margin_free': self.account.margin_free,
            'total_pnl': self.account.total_pnl,
            'total_trades': self.account.total_trades,
            'winning_trades': self.account.winning_trades,
            'losing_trades': self.account.losing_trades,
            'win_rate': win_rate,
            'return_pct': ((self.account.equity - 10000) / 10000) * 100
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """ポジション一覧取得"""
        positions = {}
        for symbol, position in self.positions.items():
            positions[symbol] = {
                'symbol': position.symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'created_at': position.created_at.isoformat()
            }
        return positions
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """注文一覧取得"""
        orders = []
        for order in self.orders.values():
            if status is None or order.status == status:
                orders.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'type': order.type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status.value,
                    'created_at': order.created_at.isoformat(),
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'filled_price': order.filled_price
                })
        return orders
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """取引履歴取得"""
        return self.trade_history[-limit:]
    
    def save_state(self):
        """状態保存"""
        try:
            state = {
                'account': asdict(self.account),
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'trade_history': self.trade_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info("Trading state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """状態復元"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    state = json.load(f)
                
                # アカウント復元
                self.account = Account(**state['account'])
                
                # ポジション復元
                self.positions = {}
                for symbol, pos_data in state['positions'].items():
                    pos_data['side'] = PositionSide(pos_data['side'])
                    pos_data['created_at'] = datetime.fromisoformat(pos_data['created_at'])
                    self.positions[symbol] = Position(**pos_data)
                
                # 取引履歴復元
                self.trade_history = state['trade_history']
                
                logger.info("Trading state loaded")
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")

# テスト用関数
def test_paper_trading():
    """ペーパートレーディングのテスト"""
    print("=== Paper Trading Engine Test ===")
    
    # エンジン初期化
    engine = PaperTradingEngine(10000.0)
    
    # 初期価格設定
    prices = {
        'BTC': 67000.0,
        'ETH': 3200.0,
        'SOL': 180.0,
        'AVAX': 30.0
    }
    engine.update_prices(prices)
    
    print("1. Initial account:")
    summary = engine.get_account_summary()
    print(f"   Balance: ${summary['balance']:,.2f}")
    print(f"   Equity: ${summary['equity']:,.2f}")
    
    # BTC購入注文
    print("\n2. Placing BTC buy order...")
    order_id = engine.place_order('BTC', OrderSide.BUY, 0.1, OrderType.MARKET)
    print(f"   Order ID: {order_id}")
    
    # ポジション確認
    positions = engine.get_positions()
    print(f"\n3. Positions: {len(positions)}")
    for symbol, pos in positions.items():
        print(f"   {symbol}: {pos['side']} {pos['quantity']} @ ${pos['entry_price']:.2f}")
    
    # 価格変動シミュレーション
    print("\n4. Price simulation...")
    new_prices = {
        'BTC': 68000.0,  # +1.49%
        'ETH': 3300.0,   # +3.13%
        'SOL': 175.0,    # -2.78%
        'AVAX': 32.0     # +6.67%
    }
    engine.update_prices(new_prices)
    
    # 更新後の状況
    summary = engine.get_account_summary()
    print(f"   New Equity: ${summary['equity']:,.2f}")
    print(f"   Total PnL: ${summary['total_pnl']:.2f}")
    
    positions = engine.get_positions()
    for symbol, pos in positions.items():
        print(f"   {symbol} Unrealized PnL: ${pos['unrealized_pnl']:.2f}")
    
    # BTC売却
    print("\n5. Selling BTC...")
    sell_order = engine.place_order('BTC', OrderSide.SELL, 0.1, OrderType.MARKET)
    
    # 最終結果
    summary = engine.get_account_summary()
    print(f"\n6. Final Results:")
    print(f"   Balance: ${summary['balance']:,.2f}")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Win Rate: {summary['win_rate']:.1f}%")
    print(f"   Return: {summary['return_pct']:+.2f}%")
    
    # 取引履歴
    trades = engine.get_trade_history()
    print(f"\n7. Trade History ({len(trades)} trades):")
    for trade in trades:
        print(f"   {trade['timestamp'][:19]} {trade['symbol']} {trade['side']} "
              f"{trade['quantity']} @ ${trade['price']:.2f} PnL: ${trade['pnl']:.2f}")

if __name__ == "__main__":
    test_paper_trading()