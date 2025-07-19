#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
プロジェクト大掃除スクリプト
不要なファイルを削除して最小限のクリーンな環境を作成
"""

import os
import shutil
from pathlib import Path

def get_essential_files():
    """残すべき重要ファイルのリスト"""
    essential_files = {
        # コアシステム
        "core/improved_prediction_system.py",
        "core/current_market_prediction.py",
        
        # ダッシュボード
        "dashboard/crypto_dashboard_fixed.py",
        "dashboard/realtime_crypto_dashboard.py",
        
        # 設定・ドキュメント
        "README.md",
        "requirements.txt",
        "requirements_dashboard.txt",
        
        # 重要なデータファイル
        "data/historical/BTC_1h_current.csv",
        "data/historical/download_summary.csv",
        "results/improved_system/improvement_summary.json",
        "results/improved_system/improvement_report.txt",
        
        # 実行ファイル
        "cleanup_project.py",
        "organize_files.py",
        
        # ユーティリティ
        "tools/historical_data_downloader.py",
        
        # 設定ファイル
        "config/config.yaml",
        
        # 重要なソースコード（最小限）
        "src/api/hyperliquid_client.py",
        "src/models/integrated_wgan_ppo.py",
        "src/features/technical_indicators.py",
        "src/data/data_loader.py"
    }
    
    return essential_files

def get_essential_directories():
    """残すべき重要ディレクトリのリスト"""
    essential_dirs = {
        "core",
        "dashboard", 
        "data/historical",
        "results/improved_system",
        "tools",
        "config",
        "src/api",
        "src/models", 
        "src/features",
        "src/data"
    }
    
    return essential_dirs

def cleanup_project():
    """プロジェクトをクリーンアップ"""
    base_dir = Path(".")
    essential_files = get_essential_files()
    essential_dirs = get_essential_directories()
    
    # 削除対象を収集
    files_to_delete = []
    dirs_to_delete = []
    
    # すべてのファイルをチェック
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        
        # 特定のディレクトリは完全スキップ
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv', 'node_modules'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = root_path / file
            relative_path = file_path.relative_to(base_dir)
            
            # 重要ファイル以外を削除対象に
            if str(relative_path).replace('\\', '/') not in essential_files:
                # ただし、一部の拡張子は保持
                if file_path.suffix not in ['.gitignore', '.env']:
                    files_to_delete.append(file_path)
    
    # 空のディレクトリや不要なディレクトリを特定
    for root, dirs, files in os.walk(base_dir, topdown=False):
        root_path = Path(root)
        relative_path = root_path.relative_to(base_dir)
        
        if str(relative_path) == '.':
            continue
            
        # 重要ディレクトリではない かつ 空になる予定の場合
        rel_path_str = str(relative_path).replace('\\', '/')
        if rel_path_str not in essential_dirs:
            # ディレクトリ内のファイルがすべて削除対象かチェック
            all_files_deleted = True
            for file_path in root_path.rglob('*'):
                if file_path.is_file():
                    relative_file = file_path.relative_to(base_dir)
                    if str(relative_file).replace('\\', '/') in essential_files:
                        all_files_deleted = False
                        break
            
            if all_files_deleted:
                dirs_to_delete.append(root_path)
    
    return files_to_delete, dirs_to_delete

def execute_cleanup(files_to_delete, dirs_to_delete, dry_run=True):
    """クリーンアップを実行"""
    print(f"削除対象ファイル数: {len(files_to_delete)}")
    print(f"削除対象ディレクトリ数: {len(dirs_to_delete)}")
    
    if dry_run:
        print("\n[ドライラン] 削除予定ファイル:")
        for file_path in files_to_delete[:20]:  # 最初の20個のみ表示
            print(f"  - {file_path}")
        
        if len(files_to_delete) > 20:
            print(f"  ... 他 {len(files_to_delete) - 20} ファイル")
        
        print("\n[ドライラン] 削除予定ディレクトリ:")
        for dir_path in dirs_to_delete[:10]:  # 最初の10個のみ表示
            print(f"  - {dir_path}")
        
        print("\n実際に削除するには dry_run=False で実行してください")
        return
    
    # 実際の削除実行
    deleted_files = 0
    deleted_dirs = 0
    
    # ファイル削除
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_files += 1
        except Exception as e:
            print(f"ファイル削除エラー {file_path}: {e}")
    
    # ディレクトリ削除
    for dir_path in dirs_to_delete:
        try:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                deleted_dirs += 1
        except Exception as e:
            print(f"ディレクトリ削除エラー {dir_path}: {e}")
    
    print(f"\n削除完了:")
    print(f"  ファイル: {deleted_files} 個")
    print(f"  ディレクトリ: {deleted_dirs} 個")

def create_clean_structure():
    """クリーンな最小構造を作成"""
    essential_dirs = [
        "core",
        "dashboard", 
        "data/historical",
        "results/improved_system",
        "tools",
        "config"
    ]
    
    for dir_path in essential_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def create_minimal_readme():
    """最小限のREADMEを作成"""
    readme_content = """# 仮想通貨AI予測システム (最小構成)

## 🚀 すぐに使える機能

### 1. リアルタイムダッシュボード (推奨)
```bash
streamlit run dashboard/realtime_crypto_dashboard.py
```
**特徴**: 自動更新、ライブ価格、リアルタイム確率表示

### 2. 固定ダッシュボード
```bash
streamlit run dashboard/crypto_dashboard_fixed.py
```

### 3. 現在市場予測
```bash
python core/current_market_prediction.py
```

### 4. システム性能テスト
```bash
python core/improved_prediction_system.py
```

## 📊 実績

- **平均予測精度**: 50.28%
- **目標達成期間**: 4/7期間 (57.1%)
- **対応銘柄**: BTC, ETH, SOL, AVAX, NEAR, ARB, OP, MATIC

## 📁 ファイル構造

```
├── core/                    # コアシステム
├── dashboard/               # WebUIダッシュボード  
├── data/historical/         # 履歴データ
├── results/improved_system/ # 性能結果
└── tools/                   # ユーティリティ
```

## ⚠️ 注意

このシステムは教育・研究目的です。実際の投資には使用しないでください。
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    """メイン実行"""
    print("プロジェクト大掃除を開始...")
    
    # クリーンアップ対象を分析
    files_to_delete, dirs_to_delete = cleanup_project()
    
    # ドライラン実行
    print("\n=== ドライラン結果 ===")
    execute_cleanup(files_to_delete, dirs_to_delete, dry_run=True)
    
    # ユーザー確認
    print("\n実際に削除を実行しますか? (y/N): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n削除を実行中...")
        execute_cleanup(files_to_delete, dirs_to_delete, dry_run=False)
        
        # クリーンな構造作成
        create_clean_structure()
        create_minimal_readme()
        
        print("\nクリーンアップ完了!")
        print("\n最小構成:")
        print("├── core/                    # メインシステム")
        print("├── dashboard/               # WebUI") 
        print("├── data/historical/         # データ")
        print("├── results/improved_system/ # 結果")
        print("└── tools/                   # ツール")
        
        print("\nすぐに使用可能:")
        print("streamlit run dashboard/realtime_crypto_dashboard.py")
        
    else:
        print("キャンセルしました")

if __name__ == "__main__":
    main()