#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
チェックポイントファイルの内容を確認するデバッグスクリプト
"""

import torch
from pathlib import Path
import sys

def debug_checkpoint(checkpoint_path):
    """チェックポイントの内容を詳細に表示"""
    print(f"\n=== Debugging checkpoint: {checkpoint_path} ===\n")
    
    try:
        # チェックポイントを読み込み
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # チェックポイントの型を確認
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # 各キーの内容を確認
            for key, value in checkpoint.items():
                if key == 'model_state_dict':
                    print(f"\n{key}:")
                    if isinstance(value, dict):
                        print(f"  Number of parameters: {len(value)}")
                        print(f"  First 5 parameter names: {list(value.keys())[:5]}")
                    else:
                        print(f"  Type: {type(value)}")
                elif key == 'optimizer_state_dict':
                    print(f"\n{key}: {'Present' if value is not None else 'None'}")
                elif key == 'metrics':
                    print(f"\n{key}: {value}")
                else:
                    print(f"\n{key}: {value}")
        else:
            print("Checkpoint is not a dictionary. It might be a raw state_dict.")
            if hasattr(checkpoint, 'keys'):
                print(f"Keys: {list(checkpoint.keys())[:10]}...")
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

def main():
    # チェックポイントディレクトリを探す
    checkpoint_dir = Path("data/models/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # すべてのチェックポイントを確認
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for i, cp in enumerate(checkpoints):
        print(f"{i+1}. {cp.name}")
    
    # 最新のgeneratorチェックポイントを確認
    generator_checkpoints = [cp for cp in checkpoints if 'generator' in cp.name]
    if generator_checkpoints:
        latest_generator = max(generator_checkpoints, key=lambda p: int(p.stem.split('_')[2]))
        print("\n" + "="*50)
        print("Checking latest GENERATOR checkpoint:")
        debug_checkpoint(latest_generator)

if __name__ == "__main__":
    main()