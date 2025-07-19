#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル整理スクリプト
プロジェクトディレクトリを整理してクリーンアップ
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """プロジェクトファイルを整理"""
    base_dir = Path(".")
    
    # 整理先ディレクトリを作成
    dirs_to_create = [
        "core",           # コアシステム
        "dashboard",      # ダッシュボード関連
        "archive/old_scripts",  # 古いスクリプト
        "archive/old_results",  # 古い結果
        "archive/experiments",  # 実験的ファイル
        "tools",          # ユーティリティツール
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # ファイル移動マッピング
    file_moves = {
        # コアシステム
        "improved_prediction_system.py": "core/",
        "current_market_prediction.py": "core/",
        
        # ダッシュボード
        "market_dashboard.py": "archive/old_scripts/",  # 旧版
        
        # 古いスクリプト
        "advanced_backtest_with_accuracy.py": "archive/old_scripts/",
        "advanced_ensemble_predictor.py": "archive/old_scripts/",
        "analyze_model_output.py": "archive/old_scripts/",
        "analyze_prediction_accuracy.py": "archive/old_scripts/",
        "backtest_analysis_report.py": "archive/old_scripts/",
        "check_feature_mismatch.py": "archive/old_scripts/",
        "check_metadata.py": "archive/old_scripts/",
        "compare_systems.py": "archive/old_scripts/",
        "comprehensive_analysis_summary.png": "archive/old_results/",
        "comprehensive_backtest.py": "archive/old_scripts/",
        
        # 実験的ファイル
        "debug_api_response.py": "archive/experiments/",
        "debug_candles.py": "archive/experiments/",
        "debug_checkpoint.py": "archive/experiments/",
        "debug_import.py": "archive/experiments/",
        "detailed_model_diagnosis.py": "archive/experiments/",
        
        # ツール類
        "historical_data_downloader.py": "tools/",
        "install_packages.py": "tools/",
        "fix_metadata.py": "tools/",
        
        # 古い結果ファイル
        "backtest_results.json": "archive/old_results/",
        "backtest_results.png": "archive/old_results/",
        "ensemble_performance_analysis.png": "archive/old_results/",
        "features_visualization.png": "archive/old_results/",
        "historical_periods_comparison.png": "archive/old_results/",
        "strategy_comparison.png": "archive/old_results/",
    }
    
    # ファイル移動実行
    moved_files = []
    for filename, destination in file_moves.items():
        source = Path(filename)
        if source.exists():
            dest_path = Path(destination)
            try:
                if source.is_file():
                    shutil.move(str(source), str(dest_path))
                    moved_files.append(f"{filename} -> {destination}")
            except Exception as e:
                print(f"移動エラー {filename}: {e}")
    
    return moved_files

def create_readme():
    """README.mdを作成"""
    readme_content = """# 仮想通貨AI予測システム

## 📁 ディレクトリ構造

```
dex-AI/
├── core/                     # コアシステム
│   ├── improved_prediction_system.py  # 改善された予測システム
│   └── current_market_prediction.py   # 現在市場予測
├── dashboard/                # ダッシュボード
│   └── crypto_dashboard_fixed.py      # 修正版ダッシュボード
├── src/                      # ソースコード
├── data/                     # データファイル
├── results/                  # 結果ファイル
├── tools/                    # ユーティリティ
├── archive/                  # アーカイブ
│   ├── old_scripts/         # 古いスクリプト
│   ├── old_results/         # 古い結果
│   └── experiments/         # 実験的コード
└── README.md                # このファイル
```

## 🚀 使用方法

### 1. 現在の市場予測
```bash
python core/current_market_prediction.py
```

### 2. ダッシュボード起動
```bash
streamlit run dashboard/crypto_dashboard_fixed.py
```

### 3. システム改善テスト
```bash
python core/improved_prediction_system.py
```

## 📊 達成結果

- **平均予測精度**: 50.28%
- **50%以上達成期間**: 4/7期間 (57.1%)
- **対応銘柄**: BTC, ETH, SOL, AVAX, NEAR, ARB, OP, MATIC

## ⚠️ 注意事項

この予測システムは教育・研究目的で開発されています。
実際の投資判断には使用しないでください。
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    """メイン実行"""
    print("プロジェクトファイルを整理中...")
    
    moved_files = organize_project()
    
    print(f"{len(moved_files)} ファイルを移動しました:")
    for move in moved_files:
        print(f"  {move}")
    
    create_readme()
    print("README.md を作成しました")
    
    print("\n整理後のディレクトリ構造:")
    print("├── core/           # コアシステム")  
    print("├── dashboard/      # ダッシュボード")
    print("├── src/            # ソースコード")
    print("├── data/           # データ")
    print("├── results/        # 結果")
    print("├── tools/          # ツール")
    print("└── archive/        # アーカイブ")
    
    print("\n主要ファイル:")
    print("core/improved_prediction_system.py     # メイン予測システム")
    print("core/current_market_prediction.py     # 現在市場予測")
    print("dashboard/crypto_dashboard_fixed.py   # WebUIダッシュボード")

if __name__ == "__main__":
    main()