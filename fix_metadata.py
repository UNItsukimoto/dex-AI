#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
メタデータを元の82特徴量に戻す
"""

import json
from pathlib import Path

def restore_metadata():
    """メタデータを元の状態に戻す"""
    
    metadata_path = Path("data/models/metadata.json")
    
    # 現在のメタデータを読み込み
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # input_sizeを82に戻す
    metadata['input_size'] = 82
    
    # 基本的な価格データを含む完全な特徴量リストを復元
    # （もしバックアップがあればそれを使用）
    backup_files = list(Path("data/models").glob("metadata.backup_*.json"))
    
    if backup_files:
        latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
        print(f"Found backup: {latest_backup}")
        
        with open(latest_backup, 'r') as f:
            backup_metadata = json.load(f)
        
        metadata['feature_columns'] = backup_metadata['feature_columns']
        metadata['input_size'] = backup_metadata['input_size']
        print(f"Restored from backup: {len(metadata['feature_columns'])} features")
    else:
        print("No backup found. Please manually restore the feature list.")
        # 基本的な特徴量を追加（最低限）
        if len(metadata['feature_columns']) == 78:
            metadata['feature_columns'] = ['open', 'high', 'low', 'close'] + metadata['feature_columns']
            metadata['input_size'] = 82
    
    # 保存
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata restored to {metadata['input_size']} features")

if __name__ == "__main__":
    restore_metadata()