#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
リアルトレーディングエンジン
Hyperliquid実取引API統合
"""

import requests
import json
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class RealTradeConfig:
    """実取引設定"""
    api_key: str
    secret_key: str
    testnet: bool = True  # Testnetから開始
    base_url: str = None
    max_position_size: float = 0.1  # 最大ポジションサイズ（10%）
    risk_limit: float = 0.05  # リスク制限（5%）
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"

class RealTradingEngine:
    """リアルトレーディングエンジン"""
    
    def __init__(self, config: RealTradeConfig):
        self.config = config
        self.session = requests.Session()
        
        # セキュリティ検証
        if not self._validate_credentials():
            raise ValueError("無効なAPI認証情報です")
        
        # 初期化ログ
        mode = "Testnet" if config.testnet else "Mainnet"
        logger.info(f"Real Trading Engine initialized - Mode: {mode}")
        
        # レート制限対応
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms間隔
    
    def _validate_credentials(self) -> bool:
        """API認証情報の検証"""
        if not self.config.api_key or not self.config.secret_key:
            logger.error("API key または Secret key が設定されていません")
            return False
        
        if len(self.config.api_key) < 32 or len(self.config.secret_key) < 32:
            logger.error("API key または Secret key の形式が無効です")
            return False
        
        return True
    
    def _create_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """HMAC署名の生成"""
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_authenticated_request(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """認証付きAPIリクエスト"""
        # レート制限対応
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        try:
            url = f"{self.config.base_url}{endpoint}"
            timestamp = str(int(time.time() * 1000))
            
            # ヘッダー作成
            headers = {
                'Content-Type': 'application/json',
                'HL-API-KEY': self.config.api_key,
                'HL-API-TIMESTAMP': timestamp,
            }
            
            # ボディ作成
            body = json.dumps(data) if data else ""
            
            # 署名作成
            signature = self._create_signature(timestamp, method, endpoint, body)
            headers['HL-API-SIGNATURE'] = signature
            
            # リクエスト実行
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """アカウント情報取得"""
        try:
            result = self._make_authenticated_request("GET", "/info/account")
            if result:
                logger.info("アカウント情報取得成功")
                return result
            return None
        except Exception as e:
            logger.error(f"アカウント情報取得エラー: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """ポジション情報取得"""
        try:
            result = self._make_authenticated_request("GET", "/info/positions")
            if result:
                return result.get('positions', [])
            return []
        except Exception as e:
            logger.error(f"ポジション取得エラー: {e}")
            return []
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: Optional[float] = None) -> Optional[str]:
        """注文実行"""
        try:
            # 注文前リスクチェック
            if not self._pre_trade_risk_check(symbol, side, quantity, price):
                logger.error("リスクチェックに失敗しました")
                return None
            
            # 注文データ作成
            order_data = {
                "coin": symbol,
                "is_buy": side == OrderSide.BUY,
                "sz": str(quantity),
                "limit_px": str(price) if price else None,
                "order_type": {
                    "limit": {"tif": "Gtc"} if order_type == OrderType.LIMIT else None,
                    "market": {} if order_type == OrderType.MARKET else None
                },
                "reduce_only": False
            }
            
            # 注文実行
            result = self._make_authenticated_request("POST", "/exchange/order", order_data)
            
            if result and result.get('status') == 'ok':
                order_id = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('resting', {}).get('oid')
                logger.info(f"注文成功: {order_id}")
                return order_id
            else:
                logger.error(f"注文失敗: {result}")
                return None
                
        except Exception as e:
            logger.error(f"注文実行エラー: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        try:
            cancel_data = {"oid": order_id}
            result = self._make_authenticated_request("POST", "/exchange/cancel", cancel_data)
            
            if result and result.get('status') == 'ok':
                logger.info(f"注文キャンセル成功: {order_id}")
                return True
            else:
                logger.error(f"注文キャンセル失敗: {result}")
                return False
                
        except Exception as e:
            logger.error(f"注文キャンセルエラー: {e}")
            return False
    
    def get_order_history(self) -> List[Dict]:
        """注文履歴取得"""
        try:
            result = self._make_authenticated_request("GET", "/info/orderHistory")
            if result:
                return result.get('orders', [])
            return []
        except Exception as e:
            logger.error(f"注文履歴取得エラー: {e}")
            return []
    
    def _pre_trade_risk_check(self, symbol: str, side: OrderSide, quantity: float, price: Optional[float]) -> bool:
        """取引前リスクチェック"""
        try:
            # 1. アカウント情報取得
            account = self.get_account_info()
            if not account:
                return False
            
            # 2. 残高チェック
            account_value = float(account.get('marginSummary', {}).get('accountValue', 0))
            if account_value <= 0:
                logger.error("アカウント残高が不足しています")
                return False
            
            # 3. ポジションサイズチェック
            estimated_value = quantity * (price or 50000)  # 概算値
            position_ratio = estimated_value / account_value
            
            if position_ratio > self.config.max_position_size:
                logger.error(f"ポジションサイズが制限を超過: {position_ratio:.2%} > {self.config.max_position_size:.2%}")
                return False
            
            # 4. 現在のポジションチェック
            positions = self.get_positions()
            current_exposure = sum(float(pos.get('szi', 0)) * float(pos.get('entryPx', 0)) for pos in positions)
            
            if current_exposure / account_value > 0.8:  # 80%以上のエクスポージャー
                logger.error("総エクスポージャーが高すぎます")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"リスクチェックエラー: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """マーケットデータ取得"""
        try:
            # 公開APIを使用（認証不要）
            url = f"{self.config.base_url}/info/allMids"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if symbol in data:
                    return {
                        'symbol': symbol,
                        'price': float(data[symbol]),
                        'timestamp': datetime.now().isoformat()
                    }
            return None
            
        except Exception as e:
            logger.error(f"マーケットデータ取得エラー: {e}")
            return None

class RealTradeConfigManager:
    """実取引設定管理"""
    
    def __init__(self, config_file: str = "real_trade_config.json"):
        self.config_file = Path(config_file)
    
    def create_config_template(self):
        """設定テンプレート作成"""
        template = {
            "api_key": "YOUR_HYPERLIQUID_API_KEY_HERE",
            "secret_key": "YOUR_HYPERLIQUID_SECRET_KEY_HERE",
            "testnet": True,
            "max_position_size": 0.1,
            "risk_limit": 0.05,
            "notes": {
                "setup_instructions": [
                    "1. Hyperliquidアカウントを作成",
                    "2. API設定でAPIキーとシークレットキーを生成",
                    "3. 上記のキーを設定ファイルに入力",
                    "4. testnet: true でテスト環境から開始",
                    "5. 十分なテスト後に testnet: false に変更"
                ],
                "security_warnings": [
                    "APIキーは絶対に他人に教えない",
                    "シークレットキーはローカルに安全に保存",
                    "設定ファイルをGitにコミットしない",
                    "定期的にキーをローテーション"
                ]
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"設定テンプレート作成: {self.config_file}")
        print(f"📝 設定ファイルが作成されました: {self.config_file}")
        print("🔑 HyperliquidのAPIキーとシークレットキーを設定してください")
    
    def load_config(self) -> Optional[RealTradeConfig]:
        """設定読み込み"""
        try:
            if not self.config_file.exists():
                print("❌ 設定ファイルが見つかりません")
                print("📝 create_config_template() を実行して設定ファイルを作成してください")
                return None
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # バリデーション
            if data.get('api_key') == "YOUR_HYPERLIQUID_API_KEY_HERE":
                print("❌ APIキーが設定されていません")
                return None
            
            if data.get('secret_key') == "YOUR_HYPERLIQUID_SECRET_KEY_HERE":
                print("❌ シークレットキーが設定されていません")
                return None
            
            config = RealTradeConfig(
                api_key=data['api_key'],
                secret_key=data['secret_key'],
                testnet=data.get('testnet', True),
                max_position_size=data.get('max_position_size', 0.1),
                risk_limit=data.get('risk_limit', 0.05)
            )
            
            logger.info("実取引設定読み込み完了")
            return config
            
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            return None

# テスト・デモ用関数
def create_demo_config():
    """デモ用設定作成"""
    config_manager = RealTradeConfigManager()
    config_manager.create_config_template()

def test_real_trading_connection():
    """実取引接続テスト"""
    config_manager = RealTradeConfigManager()
    config = config_manager.load_config()
    
    if not config:
        print("設定の読み込みに失敗しました")
        return False
    
    try:
        engine = RealTradingEngine(config)
        
        # 接続テスト
        account = engine.get_account_info()
        if account:
            print("✅ 実取引API接続成功")
            print(f"アカウント価値: ${float(account.get('marginSummary', {}).get('accountValue', 0)):,.2f}")
            return True
        else:
            print("❌ 実取引API接続失敗")
            return False
            
    except Exception as e:
        print(f"❌ 接続テストエラー: {e}")
        return False

if __name__ == "__main__":
    print("=== Real Trading Engine Demo ===")
    create_demo_config()