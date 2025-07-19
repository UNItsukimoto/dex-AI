# -*- coding: utf-8 -*-
from typing import Dict, Any

class ConfigLoader:
    """設定ファイルを管理するクラス（シンプル版）"""
    
    def __init__(self):
        self.configs = {
            'config': {
                'hyperliquid': {
                    'base_url': 'https://api.hyperliquid.xyz',
                    'ws_url': 'wss://api.hyperliquid.xyz/ws',
                    'cache_ttl': 300
                }
            }
        }
    
    def get(self, config_name: str = "config") -> Dict[str, Any]:
        """指定した設定を取得"""
        return self.configs.get(config_name, {})
    
    def get_nested(self, path: str, config_name: str = "config") -> Any:
        """ネストされた設定値を取得"""
        config = self.get(config_name)
        keys = path.split('.')
        
        for key in keys:
            if isinstance(config, dict):
                config = config.get(key)
            else:
                return None
                
        return config

# グローバルインスタンス
config = ConfigLoader()