#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量の不一致を詳細に確認
"""

import json
import pickle
from pathlib import Path

def check_feature_mismatch():
    """特徴量の不一致を確認"""
    
    # メタデータの読み込み
    with open("data/models/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # スケーラーの読み込み
    with open("data/models/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    metadata_features = metadata.get('feature_columns', [])
    scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else []
    
    print(f"Metadata features: {len(metadata_features)}")
    print(f"Scaler features: {len(scaler_features)}")
    
    # メタデータにあってスケーラーにない特徴量
    missing_in_scaler = [f for f in metadata_features if f not in scaler_features]
    print(f"\nFeatures in metadata but not in scaler ({len(missing_in_scaler)}):")
    for f in missing_in_scaler:
        print(f"  - {f}")
    
    # スケーラーにあってメタデータにない特徴量
    missing_in_metadata = [f for f in scaler_features if f not in metadata_features]
    print(f"\nFeatures in scaler but not in metadata ({len(missing_in_metadata)}):")
    for f in missing_in_metadata:
        print(f"  - {f}")
    
    # 共通の特徴量
    common_features = [f for f in metadata_features if f in scaler_features]
    print(f"\nCommon features: {len(common_features)}")
    
    # 推奨される解決策
    print("\n=== Recommended Solution ===")
    print("Use only the features that the scaler was trained on:")
    print("1. Update metadata to use scaler's feature list")
    print("2. Or retrain the model with consistent features")
    
    return scaler_features, metadata_features

if __name__ == "__main__":
    scaler_features, metadata_features = check_feature_mismatch()