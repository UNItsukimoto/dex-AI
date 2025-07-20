#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合エラーハンドリングシステム
アプリケーション全体の例外処理を統一管理
"""

import logging
import traceback
import streamlit as st
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_platform.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingPlatformError(Exception):
    """取引プラットフォーム基底例外"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
        self.timestamp = datetime.now()

class APIConnectionError(TradingPlatformError):
    """API接続エラー"""
    def __init__(self, message: str, api_name: str = None):
        super().__init__(message, 'API_CONNECTION_ERROR', {'api_name': api_name})

class DataValidationError(TradingPlatformError):
    """データ検証エラー"""
    def __init__(self, message: str, field_name: str = None, value: Any = None):
        super().__init__(message, 'DATA_VALIDATION_ERROR', {
            'field_name': field_name,
            'value': str(value) if value is not None else None
        })

class TradingError(TradingPlatformError):
    """取引実行エラー"""
    def __init__(self, message: str, symbol: str = None, order_type: str = None):
        super().__init__(message, 'TRADING_ERROR', {
            'symbol': symbol,
            'order_type': order_type
        })

class PredictionError(TradingPlatformError):
    """予測エラー"""
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, 'PREDICTION_ERROR', {'model_name': model_name})

class ErrorHandler:
    """統合エラーハンドラー"""
    
    def __init__(self):
        self.error_history = []
        self.max_history = 100
    
    def handle_error(self, error: Exception, context: str = None, show_user: bool = True) -> Dict:
        """エラーを処理し、ログ記録とユーザー通知を行う"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # カスタム例外の詳細情報追加
        if isinstance(error, TradingPlatformError):
            error_info.update({
                'error_code': error.error_code,
                'details': error.details
            })
        
        # ログ記録
        logger.error(f"Error in {context}: {error_info}")
        
        # エラー履歴に追加
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # ユーザー通知
        if show_user:
            self._show_user_error(error, context)
        
        return error_info
    
    def _show_user_error(self, error: Exception, context: str = None):
        """ユーザーにエラーを表示"""
        if isinstance(error, APIConnectionError):
            st.error(f"🌐 API接続エラー: {error.message}")
            st.info("💡 ネットワーク接続を確認するか、しばらく待ってから再試行してください。")
            
        elif isinstance(error, DataValidationError):
            st.error(f"📊 データエラー: {error.message}")
            if error.details.get('field_name'):
                st.info(f"💡 フィールド '{error.details['field_name']}' の値を確認してください。")
                
        elif isinstance(error, TradingError):
            st.error(f"💰 取引エラー: {error.message}")
            st.info("💡 注文パラメータを確認するか、残高をチェックしてください。")
            
        elif isinstance(error, PredictionError):
            st.error(f"🤖 予測エラー: {error.message}")
            st.info("💡 データが不足しているか、モデルの再学習が必要な可能性があります。")
            
        else:
            st.error(f"❌ システムエラー: {str(error)}")
            if context:
                st.info(f"💡 エラー発生箇所: {context}")
    
    def get_error_summary(self) -> Dict:
        """エラーサマリーを取得"""
        if not self.error_history:
            return {'total_errors': 0, 'recent_errors': []}
        
        recent_errors = self.error_history[-10:]  # 最新10件
        error_types = {}
        
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': recent_errors,
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }

# グローバルエラーハンドラーインスタンス
error_handler = ErrorHandler()

def safe_execute(context: str = None, show_error: bool = True, default_return: Any = None):
    """関数を安全に実行するデコレータ"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context or func.__name__, show_error)
                return default_return
        return wrapper
    return decorator

def safe_get(data: Dict, key: str, default: Any = None, required: bool = False) -> Any:
    """安全な辞書アクセス（バリデーション付き）"""
    try:
        if key not in data:
            if required:
                raise DataValidationError(f"必須フィールド '{key}' が見つかりません", key)
            return default
        
        value = data[key]
        if value is None and required:
            raise DataValidationError(f"フィールド '{key}' の値がNullです", key, value)
        
        return value if value is not None else default
        
    except Exception as e:
        if not isinstance(e, DataValidationError):
            error_handler.handle_error(e, f"safe_get: {key}")
        raise

def validate_numeric(value: Any, field_name: str, min_val: float = None, max_val: float = None) -> float:
    """数値バリデーション"""
    try:
        if value is None:
            raise DataValidationError(f"数値フィールド '{field_name}' がNullです", field_name, value)
        
        numeric_value = float(value)
        
        if min_val is not None and numeric_value < min_val:
            raise DataValidationError(f"'{field_name}' は {min_val} 以上である必要があります", field_name, value)
        
        if max_val is not None and numeric_value > max_val:
            raise DataValidationError(f"'{field_name}' は {max_val} 以下である必要があります", field_name, value)
        
        return numeric_value
        
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"'{field_name}' は有効な数値ではありません", field_name, value)

def validate_symbol(symbol: str) -> str:
    """銘柄バリデーション"""
    if not symbol or not isinstance(symbol, str):
        raise DataValidationError("銘柄コードが無効です", 'symbol', symbol)
    
    symbol = symbol.upper().strip()
    
    # 基本的な銘柄コード形式チェック
    if not symbol.isalnum() or len(symbol) < 2 or len(symbol) > 10:
        raise DataValidationError("銘柄コード形式が無効です", 'symbol', symbol)
    
    return symbol

@safe_execute("error_display", show_error=False)
def display_error_dashboard():
    """エラー状況ダッシュボード"""
    with st.expander("🔍 システム状態", expanded=False):
        summary = error_handler.get_error_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("総エラー数", summary['total_errors'])
        
        with col2:
            if summary['most_common_error']:
                st.metric("最多エラー型", summary['most_common_error'])
        
        with col3:
            recent_count = len(summary['recent_errors'])
            st.metric("直近エラー", recent_count)
        
        if summary['recent_errors']:
            st.subheader("最近のエラー")
            for error in summary['recent_errors'][-5:]:  # 最新5件のみ表示
                with st.container():
                    st.write(f"**{error['timestamp'].strftime('%H:%M:%S')}** - {error['error_type']}")
                    if error.get('context'):
                        st.write(f"場所: {error['context']}")
                    st.write(f"メッセージ: {error['message']}")
                    st.divider()

def create_error_boundary(component_name: str):
    """Streamlitコンポーネント用エラーバウンダリー"""
    def error_boundary_decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, f"{component_name}_component")
                
                # フォールバック表示
                with st.container():
                    st.error(f"⚠️ {component_name}コンポーネントでエラーが発生しました")
                    st.info("システム管理者に連絡するか、ページを再読み込みしてください")
                    
                    with st.expander("詳細情報"):
                        st.code(str(e))
                
                return None
        return wrapper
    return error_boundary_decorator

# Streamlit固有のエラーハンドリング
def init_streamlit_error_handling():
    """Streamlitのエラーハンドリングを初期化"""
    
    # ページ例外ハンドラー
    def handle_streamlit_error(error):
        error_handler.handle_error(error, "streamlit_page")
    
    # セッション状態のエラーチェック
    if 'error_handler_initialized' not in st.session_state:
        st.session_state.error_handler_initialized = True
        logger.info("Streamlit error handling initialized")

if __name__ == "__main__":
    # テスト実行
    print("=== Error Handler Test ===")
    
    # テスト用エラー
    try:
        raise APIConnectionError("テスト用API接続エラー", "HyperLiquid")
    except Exception as e:
        error_handler.handle_error(e, "test_context", show_user=False)
    
    try:
        raise DataValidationError("テスト用データ検証エラー", "price", "invalid_price")
    except Exception as e:
        error_handler.handle_error(e, "test_context", show_user=False)
    
    # サマリー表示
    summary = error_handler.get_error_summary()
    print(f"Total errors: {summary['total_errors']}")
    print(f"Error types: {summary['error_types']}")