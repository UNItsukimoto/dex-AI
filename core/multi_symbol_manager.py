#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マルチ銘柄管理システム
複数暗号通貨の同時監視・取引機能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
import asyncio
import concurrent.futures
from dataclasses import dataclass
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SymbolConfig:
    """銘柄設定クラス"""
    symbol: str
    enabled: bool = True
    max_position_size: float = 0.1  # ポートフォリオの10%
    min_confidence: float = 0.75    # 最小信頼度
    stop_loss_pct: float = 0.05     # 5%ストップロス
    take_profit_pct: float = 0.15   # 15%利確
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'enabled': self.enabled,
            'max_position_size': self.max_position_size,
            'min_confidence': self.min_confidence,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

class MultiSymbolManager:
    """マルチ銘柄管理システム"""
    
    def __init__(self, config_file: str = "multi_symbol_config.json"):
        self.config_file = Path(config_file)
        
        # デフォルト対応銘柄
        self.default_symbols = {
            'BTC': SymbolConfig('BTC', True, 0.15, 0.8, 0.05, 0.12),
            'ETH': SymbolConfig('ETH', True, 0.12, 0.75, 0.06, 0.15),
            'SOL': SymbolConfig('SOL', True, 0.08, 0.7, 0.08, 0.18),
            'AVAX': SymbolConfig('AVAX', False, 0.06, 0.7, 0.08, 0.18),
            'DOGE': SymbolConfig('DOGE', False, 0.05, 0.65, 0.1, 0.2),
            'MATIC': SymbolConfig('MATIC', False, 0.05, 0.65, 0.1, 0.2),
            'NEAR': SymbolConfig('NEAR', False, 0.04, 0.65, 0.1, 0.2),
            'APT': SymbolConfig('APT', False, 0.04, 0.65, 0.1, 0.2),
            'ARB': SymbolConfig('ARB', False, 0.04, 0.65, 0.1, 0.2),
            'OP': SymbolConfig('OP', False, 0.04, 0.65, 0.1, 0.2),
            'SUI': SymbolConfig('SUI', False, 0.03, 0.6, 0.12, 0.22),
            'SEI': SymbolConfig('SEI', False, 0.03, 0.6, 0.12, 0.22)
        }
        
        # 設定読み込み
        self.symbol_configs = self._load_config()
        
        # ランタイムデータ
        self.live_prices = {}
        self.predictions = {}
        self.last_update = {}
        self.correlation_matrix = pd.DataFrame()
        
        logger.info(f"マルチ銘柄管理システム初期化: {len(self.get_enabled_symbols())}銘柄有効")
    
    def _load_config(self) -> Dict[str, SymbolConfig]:
        """設定ファイル読み込み"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                configs = {}
                for symbol, data in config_data.items():
                    configs[symbol] = SymbolConfig(
                        symbol=data['symbol'],
                        enabled=data.get('enabled', True),
                        max_position_size=data.get('max_position_size', 0.1),
                        min_confidence=data.get('min_confidence', 0.75),
                        stop_loss_pct=data.get('stop_loss_pct', 0.05),
                        take_profit_pct=data.get('take_profit_pct', 0.15)
                    )
                
                logger.info(f"設定ファイル読み込み完了: {len(configs)}銘柄")
                return configs
                
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}. デフォルト設定を使用")
        
        return self.default_symbols.copy()
    
    def save_config(self):
        """設定ファイル保存"""
        try:
            config_data = {}
            for symbol, config in self.symbol_configs.items():
                config_data[symbol] = config.to_dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
            logger.info("設定ファイル保存完了")
            
        except Exception as e:
            logger.error(f"設定ファイル保存エラー: {e}")
    
    def get_enabled_symbols(self) -> List[str]:
        """有効な銘柄リスト取得"""
        return [symbol for symbol, config in self.symbol_configs.items() if config.enabled]
    
    def get_all_symbols(self) -> List[str]:
        """全銘柄リスト取得"""
        return list(self.symbol_configs.keys())
    
    def enable_symbol(self, symbol: str):
        """銘柄を有効化"""
        if symbol in self.symbol_configs:
            self.symbol_configs[symbol].enabled = True
            logger.info(f"{symbol} 有効化")
        else:
            logger.warning(f"{symbol} は設定に存在しません")
    
    def disable_symbol(self, symbol: str):
        """銘柄を無効化"""
        if symbol in self.symbol_configs:
            self.symbol_configs[symbol].enabled = False
            logger.info(f"{symbol} 無効化")
        else:
            logger.warning(f"{symbol} は設定に存在しません")
    
    def update_symbol_config(self, symbol: str, **kwargs):
        """銘柄設定更新"""
        if symbol not in self.symbol_configs:
            logger.warning(f"{symbol} は設定に存在しません")
            return
        
        config = self.symbol_configs[symbol]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"{symbol} {key} を {value} に更新")
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """銘柄設定取得"""
        return self.symbol_configs.get(symbol)
    
    def update_live_data(self, symbol: str, price: float, prediction: Dict):
        """ライブデータ更新"""
        self.live_prices[symbol] = price
        self.predictions[symbol] = prediction
        self.last_update[symbol] = datetime.now()
    
    def get_trading_opportunities(self) -> List[Dict]:
        """取引機会の分析"""
        opportunities = []
        
        for symbol in self.get_enabled_symbols():
            config = self.symbol_configs[symbol]
            prediction = self.predictions.get(symbol, {})
            
            if not prediction:
                continue
            
            confidence = prediction.get('confidence', 0)
            signal = prediction.get('signal', 'HOLD')
            
            # 最小信頼度チェック
            if confidence >= config.min_confidence and signal != 'HOLD':
                opportunity = {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'price': self.live_prices.get(symbol, 0),
                    'max_position_size': config.max_position_size,
                    'stop_loss_pct': config.stop_loss_pct,
                    'take_profit_pct': config.take_profit_pct,
                    'timestamp': datetime.now()
                }
                opportunities.append(opportunity)
        
        # 信頼度でソート
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """銘柄間相関計算"""
        try:
            enabled_symbols = self.get_enabled_symbols()
            if len(enabled_symbols) < 2:
                return pd.DataFrame()
            
            # 価格データのダミー作成（実際の実装では履歴データを使用）
            price_data = {}
            for symbol in enabled_symbols:
                # ダミーデータ（実際の実装ではAPIから取得）
                price_data[symbol] = np.random.normal(0, 1, 100).cumsum()
            
            df = pd.DataFrame(price_data)
            correlation_matrix = df.corr()
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"相関計算エラー: {e}")
            return pd.DataFrame()
    
    def get_portfolio_exposure(self, current_positions: Dict[str, float]) -> Dict:
        """ポートフォリオエクスポージャー分析"""
        total_exposure = sum(abs(pos) for pos in current_positions.values())
        
        exposure_data = {
            'total_exposure': total_exposure,
            'symbol_exposures': current_positions,
            'diversification_score': self._calculate_diversification_score(current_positions),
            'correlation_risk': self._calculate_correlation_risk(current_positions)
        }
        
        return exposure_data
    
    def _calculate_diversification_score(self, positions: Dict[str, float]) -> float:
        """分散度スコア計算"""
        if not positions:
            return 1.0
        
        total_exposure = sum(abs(pos) for pos in positions.values())
        if total_exposure == 0:
            return 1.0
        
        # ハーフィンダール指数を使用
        weights = [abs(pos) / total_exposure for pos in positions.values()]
        hhi = sum(w**2 for w in weights)
        
        # 0-1スコアに正規化（1が最も分散）
        max_hhi = 1.0  # 1銘柄のみの場合
        return 1.0 - hhi
    
    def _calculate_correlation_risk(self, positions: Dict[str, float]) -> float:
        """相関リスク計算"""
        if len(positions) < 2 or self.correlation_matrix.empty:
            return 0.0
        
        try:
            symbols = list(positions.keys())
            total_exposure = sum(abs(pos) for pos in positions.values())
            
            if total_exposure == 0:
                return 0.0
            
            correlation_risk = 0.0
            count = 0
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                        corr = self.correlation_matrix.loc[symbol1, symbol2]
                        weight1 = abs(positions[symbol1]) / total_exposure
                        weight2 = abs(positions[symbol2]) / total_exposure
                        
                        correlation_risk += abs(corr) * weight1 * weight2
                        count += 1
            
            return correlation_risk / max(count, 1)
            
        except Exception as e:
            logger.error(f"相関リスク計算エラー: {e}")
            return 0.0
    
    def get_symbol_performance_metrics(self) -> Dict[str, Dict]:
        """銘柄別パフォーマンス指標"""
        metrics = {}
        
        for symbol in self.get_enabled_symbols():
            prediction = self.predictions.get(symbol, {})
            price = self.live_prices.get(symbol, 0)
            last_update = self.last_update.get(symbol)
            
            metrics[symbol] = {
                'current_price': price,
                'prediction_confidence': prediction.get('confidence', 0),
                'prediction_signal': prediction.get('signal', 'UNKNOWN'),
                'last_update': last_update.isoformat() if last_update else None,
                'data_freshness': (datetime.now() - last_update).total_seconds() if last_update else float('inf')
            }
        
        return metrics
    
    def generate_trading_summary(self) -> Dict:
        """取引サマリー生成"""
        enabled_symbols = self.get_enabled_symbols()
        opportunities = self.get_trading_opportunities()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(self.symbol_configs),
            'enabled_symbols': len(enabled_symbols),
            'trading_opportunities': len(opportunities),
            'high_confidence_signals': len([op for op in opportunities if op['confidence'] >= 0.85]),
            'buy_signals': len([op for op in opportunities if op['signal'] == 'BUY']),
            'sell_signals': len([op for op in opportunities if op['signal'] == 'SELL']),
            'enabled_symbol_list': enabled_symbols,
            'top_opportunities': opportunities[:5]  # トップ5機会
        }
        
        return summary

    def export_config_template(self, filename: str = "multi_symbol_config_template.json"):
        """設定テンプレート出力"""
        try:
            template = {}
            for symbol, config in self.default_symbols.items():
                template[symbol] = config.to_dict()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
                
            logger.info(f"設定テンプレート出力完了: {filename}")
            
        except Exception as e:
            logger.error(f"テンプレート出力エラー: {e}")