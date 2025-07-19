#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
メタデータとスケーラーの情報を確認
"""

import json
import pickle
from pathlib import Path

def check_metadata():
    """メタデータとスケーラーの情報を表示"""
    
    # メタデータの確認
    metadata_path = Path("data/models/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("=== Metadata ===")
        print(f"Symbol: {metadata.get('symbol')}")
        print(f"Epochs: {metadata.get('epochs')}")
        print(f"Input size: {metadata.get('input_size')}")
        print(f"Sequence length: {metadata.get('sequence_length')}")
        print(f"Number of features: {len(metadata.get('feature_columns', []))}")
        print(f"\nFirst 10 features:")
        for i, col in enumerate(metadata.get('feature_columns', [])[:10]):
            print(f"  {i+1}. {col}")
    else:
        print("Metadata file not found")
    
    # スケーラーの確認
    scaler_path = Path("data/models/scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("\n=== Scaler Info ===")
        print(f"Scaler type: {type(scaler).__name__}")
        if hasattr(scaler, 'n_features_in_'):
            print(f"Expected number of features: {scaler.n_features_in_}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Feature names available: {'Yes' if scaler.feature_names_in_ is not None else 'No'}")
            if scaler.feature_names_in_ is not None:
                print(f"Number of feature names: {len(scaler.feature_names_in_)}")
    else:
        print("Scaler file not found")

if __name__ == "__main__":
    check_metadata()