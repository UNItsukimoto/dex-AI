#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
セキュアな設定管理システム
APIキー、秘密鍵等の機密情報を安全に管理
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import streamlit as st

# 暗号化ライブラリ（オプション）
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # フォールバック用の簡易暗号化
    import base64

logger = logging.getLogger(__name__)

class SecureConfig:
    """セキュアな設定管理クラス"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.hyperliquid_ai'
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self.config_file = self.config_dir / 'config.enc'
        self.key_file = self.config_dir / '.key'
        
        # 暗号化キーの初期化
        self._init_encryption_key()
        
        # デフォルト設定
        self.default_config = {
            'api_settings': {
                'hyperliquid_api_key': '',
                'hyperliquid_secret_key': '',
                'testnet': True,
                'timeout': 30,
                'max_retries': 3
            },
            'trading_settings': {
                'max_position_size': 1000,
                'default_leverage': 1,
                'risk_limit_per_trade': 0.02,  # 2%
                'stop_loss_default': 0.05,    # 5%
                'take_profit_default': 0.10   # 10%
            },
            'ui_settings': {
                'theme': 'dark',
                'default_symbols': ['BTC', 'ETH', 'SOL'],
                'refresh_interval': 5,  # seconds
                'show_advanced_features': False
            },
            'security_settings': {
                'session_timeout': 3600,  # 1 hour
                'require_password': False,
                'enable_2fa': False,
                'log_level': 'INFO'
            }
        }
        
        # 設定をロード
        self.config = self._load_config()
    
    def _init_encryption_key(self):
        """暗号化キーの初期化"""
        try:
            if CRYPTO_AVAILABLE:
                # cryptographyライブラリ使用
                if self.key_file.exists():
                    # 既存のキーを読み込み
                    with open(self.key_file, 'rb') as f:
                        key_data = f.read()
                    self.fernet = Fernet(key_data)
                else:
                    # 新しいキーを生成
                    password = os.urandom(32)  # ランダムパスワード生成
                    salt = os.urandom(16)
                    
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=salt,
                        iterations=100000,
                    )
                    
                    key = base64.urlsafe_b64encode(kdf.derive(password))
                    self.fernet = Fernet(key)
                    
                    # キーを保存（権限を制限）
                    with open(self.key_file, 'wb') as f:
                        f.write(key)
                    
                    # ファイル権限を所有者のみに制限
                    try:
                        os.chmod(self.key_file, 0o600)
                    except OSError:
                        # Windowsでは権限設定をスキップ
                        pass
                    
                    logger.info("New encryption key generated")
            else:
                # フォールバック：base64エンコーディングのみ
                logger.warning("cryptography library not available, using basic encoding")
                self.fernet = None
                
        except Exception as e:
            logger.error(f"Encryption key initialization failed: {e}")
            # エラー時はフォールバックモード
            self.fernet = None
    
    def _encrypt_data(self, data: Dict) -> bytes:
        """データを暗号化"""
        try:
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            if self.fernet and CRYPTO_AVAILABLE:
                # 強力な暗号化
                encrypted_data = self.fernet.encrypt(json_data)
            else:
                # フォールバック：base64エンコーディング
                encrypted_data = base64.b64encode(json_data)
                
            return encrypted_data
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict:
        """データを復号化"""
        try:
            if self.fernet and CRYPTO_AVAILABLE:
                # 強力な復号化
                decrypted_data = self.fernet.decrypt(encrypted_data)
            else:
                # フォールバック：base64デコーディング
                decrypted_data = base64.b64decode(encrypted_data)
                
            json_data = decrypted_data.decode('utf-8')
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise
    
    def _load_config(self) -> Dict:
        """設定をロード"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'rb') as f:
                    encrypted_data = f.read()
                
                if encrypted_data:
                    config = self._decrypt_data(encrypted_data)
                    # デフォルト設定とマージ
                    return self._merge_config(self.default_config, config)
            
            # 設定ファイルが存在しない場合はデフォルトを返す
            logger.info("No config file found, using defaults")
            return self.default_config.copy()
            
        except Exception as e:
            logger.error(f"Config loading failed: {e}")
            # エラー時はデフォルト設定を返す
            return self.default_config.copy()
    
    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """設定をマージ（デフォルト + ユーザー設定）"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self):
        """設定を保存"""
        try:
            encrypted_data = self._encrypt_data(self.config)
            
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_data)
            
            # ファイル権限を所有者のみに制限
            try:
                os.chmod(self.config_file, 0o600)
            except OSError:
                # Windowsでは権限設定をスキップ
                pass
            
            logger.info("Config saved successfully")
            
        except Exception as e:
            logger.error(f"Config saving failed: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法対応）"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Config get failed for key '{key_path}': {e}")
            return default
    
    def set(self, key_path: str, value: Any):
        """設定値を設定（ドット記法対応）"""
        try:
            keys = key_path.split('.')
            config = self.config
            
            # 最後のキー以外をたどる
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # 最後のキーに値を設定
            config[keys[-1]] = value
            
            logger.info(f"Config set: {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"Config set failed for key '{key_path}': {e}")
            raise
    
    def has_api_credentials(self) -> bool:
        """API認証情報が設定されているかチェック"""
        api_key = self.get('api_settings.hyperliquid_api_key', '').strip()
        secret_key = self.get('api_settings.hyperliquid_secret_key', '').strip()
        
        return bool(api_key and secret_key)
    
    def validate_config(self) -> Dict[str, str]:
        """設定を検証"""
        errors = {}
        
        # API設定検証
        if not self.has_api_credentials() and not self.get('api_settings.testnet', True):
            errors['api_credentials'] = 'メインネット使用時はAPI認証情報が必要です'
        
        # 取引設定検証
        max_position = self.get('trading_settings.max_position_size', 0)
        if max_position <= 0:
            errors['max_position_size'] = '最大ポジションサイズは正の値である必要があります'
        
        risk_limit = self.get('trading_settings.risk_limit_per_trade', 0)
        if not 0 < risk_limit <= 1:
            errors['risk_limit'] = 'リスク制限は0〜1の範囲で設定してください'
        
        # UI設定検証
        refresh_interval = self.get('ui_settings.refresh_interval', 0)
        if refresh_interval < 1:
            errors['refresh_interval'] = '更新間隔は1秒以上である必要があります'
        
        return errors
    
    def reset_to_defaults(self):
        """設定をデフォルトにリセット"""
        self.config = self.default_config.copy()
        logger.info("Config reset to defaults")
    
    def export_config(self, include_secrets: bool = False) -> Dict:
        """設定をエクスポート"""
        config_copy = self.config.copy()
        
        if not include_secrets:
            # 機密情報を除外
            if 'api_settings' in config_copy:
                config_copy['api_settings'].pop('hyperliquid_api_key', None)
                config_copy['api_settings'].pop('hyperliquid_secret_key', None)
        
        return config_copy
    
    def get_masked_api_key(self) -> str:
        """マスクされたAPIキーを取得"""
        api_key = self.get('api_settings.hyperliquid_api_key', '')
        if len(api_key) > 8:
            return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
        return '*' * len(api_key) if api_key else ''

# グローバル設定インスタンス
config = SecureConfig()

def init_config_ui():
    """設定UIを初期化"""
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = config

def create_settings_panel():
    """設定パネルを作成"""
    st.subheader("⚙️ システム設定")
    
    # 暗号化ライブラリの状態表示
    if not CRYPTO_AVAILABLE:
        st.warning("⚠️ 暗号化ライブラリが利用できません。基本的なエンコーディングを使用します。")
        st.info("セキュリティを向上させるには `pip install cryptography` を実行してください。")
    else:
        st.success("🔒 強力な暗号化が有効です")
    
    # 設定タブ
    tab1, tab2, tab3, tab4 = st.tabs(["API設定", "取引設定", "UI設定", "セキュリティ"])
    
    with tab1:
        st.write("**API接続設定**")
        
        # テストネット切り替え
        testnet = st.checkbox(
            "テストネットを使用",
            value=config.get('api_settings.testnet', True),
            help="本番環境での取引を行う場合はチェックを外してください"
        )
        config.set('api_settings.testnet', testnet)
        
        if not testnet:
            st.warning("⚠️ 本番環境での取引は実際の資金が必要です")
        
        # API認証情報
        if st.checkbox("API認証情報を設定", value=config.has_api_credentials()):
            api_key = st.text_input(
                "APIキー",
                value=config.get('api_settings.hyperliquid_api_key', ''),
                type="password",
                help="HyperLiquid APIキー"
            )
            
            secret_key = st.text_input(
                "シークレットキー",
                value=config.get('api_settings.hyperliquid_secret_key', ''),
                type="password",
                help="HyperLiquid シークレットキー"
            )
            
            if api_key:
                config.set('api_settings.hyperliquid_api_key', api_key)
            if secret_key:
                config.set('api_settings.hyperliquid_secret_key', secret_key)
        
        # 接続設定
        timeout = st.slider(
            "タイムアウト (秒)",
            min_value=5,
            max_value=60,
            value=config.get('api_settings.timeout', 30)
        )
        config.set('api_settings.timeout', timeout)
    
    with tab2:
        st.write("**取引リスク設定**")
        
        max_position = st.number_input(
            "最大ポジションサイズ (USD)",
            min_value=100,
            max_value=100000,
            value=config.get('trading_settings.max_position_size', 1000)
        )
        config.set('trading_settings.max_position_size', max_position)
        
        risk_limit = st.slider(
            "1取引あたりのリスク制限 (%)",
            min_value=0.1,
            max_value=10.0,
            value=config.get('trading_settings.risk_limit_per_trade', 2.0),
            format="%.1f%%"
        )
        config.set('trading_settings.risk_limit_per_trade', risk_limit / 100)
        
        stop_loss = st.slider(
            "デフォルトストップロス (%)",
            min_value=1.0,
            max_value=20.0,
            value=config.get('trading_settings.stop_loss_default', 5.0) * 100,
            format="%.1f%%"
        )
        config.set('trading_settings.stop_loss_default', stop_loss / 100)
    
    with tab3:
        st.write("**ユーザーインターフェース設定**")
        
        theme = st.selectbox(
            "テーマ",
            options=["dark", "light"],
            index=0 if config.get('ui_settings.theme', 'dark') == 'dark' else 1
        )
        config.set('ui_settings.theme', theme)
        
        refresh_interval = st.slider(
            "データ更新間隔 (秒)",
            min_value=1,
            max_value=30,
            value=config.get('ui_settings.refresh_interval', 5)
        )
        config.set('ui_settings.refresh_interval', refresh_interval)
        
        show_advanced = st.checkbox(
            "高度な機能を表示",
            value=config.get('ui_settings.show_advanced_features', False)
        )
        config.set('ui_settings.show_advanced_features', show_advanced)
    
    with tab4:
        st.write("**セキュリティ設定**")
        
        session_timeout = st.slider(
            "セッションタイムアウト (分)",
            min_value=15,
            max_value=480,
            value=config.get('security_settings.session_timeout', 60)
        )
        config.set('security_settings.session_timeout', session_timeout * 60)
        
        log_level = st.selectbox(
            "ログレベル",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1  # INFO
        )
        config.set('security_settings.log_level', log_level)
        
        # 設定の保存
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 設定を保存"):
                try:
                    errors = config.validate_config()
                    if errors:
                        for field, error in errors.items():
                            st.error(f"{field}: {error}")
                    else:
                        config.save_config()
                        st.success("設定が保存されました")
                except Exception as e:
                    st.error(f"設定保存エラー: {e}")
        
        with col2:
            if st.button("🔄 デフォルトに戻す"):
                config.reset_to_defaults()
                st.success("設定をデフォルトに戻しました")
                st.experimental_rerun()
        
        with col3:
            if st.button("📤 設定をエクスポート"):
                exported = config.export_config(include_secrets=False)
                st.download_button(
                    label="📥 ダウンロード",
                    data=json.dumps(exported, indent=2, ensure_ascii=False),
                    file_name=f"hyperliquid_ai_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    # テスト実行
    print("=== Secure Config Test ===")
    
    test_config = SecureConfig()
    
    # 設定テスト
    test_config.set('api_settings.hyperliquid_api_key', 'test_key_123')
    test_config.set('trading_settings.max_position_size', 5000)
    
    print(f"API Key (masked): {test_config.get_masked_api_key()}")
    print(f"Max position: {test_config.get('trading_settings.max_position_size')}")
    print(f"Has credentials: {test_config.has_api_credentials()}")
    
    # 設定保存テスト
    test_config.save_config()
    print("Config saved successfully")
    
    # 検証テスト
    errors = test_config.validate_config()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Config validation passed")