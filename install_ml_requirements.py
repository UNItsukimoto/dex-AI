#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
機械学習ライブラリ自動インストール
予測精度向上のための追加パッケージインストール
"""

import subprocess
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_install_package(package_name, import_name=None):
    """パッケージの確認とインストール"""
    if import_name is None:
        import_name = package_name
    
    try:
        # インポート試行
        importlib.import_module(import_name)
        logger.info(f"[OK] {package_name} はすでにインストールされています")
        return True
    except ImportError:
        logger.info(f"[警告] {package_name} がインストールされていません。インストールを開始します...")
        
        try:
            # パッケージインストール
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, "--quiet"
            ])
            
            # インストール確認
            importlib.import_module(import_name)
            logger.info(f"[完了] {package_name} のインストールが完了しました")
            return True
            
        except Exception as e:
            logger.error(f"[エラー] {package_name} のインストールに失敗しました: {e}")
            return False

def install_ml_requirements():
    """機械学習ライブラリの一括インストール"""
    logger.info("=== 機械学習ライブラリ自動インストール開始 ===")
    
    # 必要なパッケージリスト
    packages = [
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("scipy", "scipy"),
    ]
    
    # インストール状況追跡
    installation_results = {}
    
    for package_name, import_name in packages:
        result = check_and_install_package(package_name, import_name)
        installation_results[package_name] = result
    
    # 結果サマリー
    logger.info("\n=== インストール結果サマリー ===")
    successful = 0
    for package, success in installation_results.items():
        status = "[成功]" if success else "[失敗]"
        logger.info(f"{package}: {status}")
        if success:
            successful += 1
    
    logger.info(f"\n{successful}/{len(packages)} パッケージのインストールが完了しました")
    
    if successful == len(packages):
        logger.info("[完了] すべての機械学習ライブラリが利用可能です！")
        logger.info("高度予測エンジンがフル機能で動作します。")
    elif successful > 0:
        logger.info("[部分] 一部の機械学習ライブラリが利用可能です。")
        logger.info("基本的な機械学習機能は動作しますが、一部機能が制限される可能性があります。")
    else:
        logger.warning("[制限] 機械学習ライブラリが利用できません。")
        logger.warning("従来の予測手法のみで動作します。")
    
    return installation_results

def test_ml_functionality():
    """機械学習機能のテスト"""
    logger.info("\n=== 機械学習機能テスト開始 ===")
    
    try:
        from core.advanced_prediction_engine import AdvancedPredictionEngine
        
        # エンジン初期化テスト
        engine = AdvancedPredictionEngine()
        logger.info("[OK] 高度予測エンジンの初期化成功")
        
        # 性能情報取得テスト
        performance = engine.get_model_performance()
        ml_available = performance.get('ml_available', False)
        
        if ml_available:
            logger.info("[OK] 機械学習機能が正常に動作しています")
            logger.info(f"学習サンプル数: {performance.get('training_samples', 0)}")
            logger.info(f"予測実行回数: {performance.get('prediction_count', 0)}")
        else:
            logger.info("[注意] 機械学習機能は利用できませんが、基本機能は動作します")
        
        return True
        
    except Exception as e:
        logger.error(f"[エラー] 機械学習機能テストに失敗: {e}")
        return False

def main():
    """メイン実行"""
    print("AI予測精度向上システム セットアップ")
    print("=" * 50)
    
    # 機械学習ライブラリインストール
    installation_results = install_ml_requirements()
    
    # 機能テスト
    test_result = test_ml_functionality()
    
    print("\n" + "=" * 50)
    print("セットアップ完了サマリー")
    print("=" * 50)
    
    # scikit-learnの確認
    sklearn_available = installation_results.get('scikit-learn', False)
    xgboost_available = installation_results.get('xgboost', False)
    lightgbm_available = installation_results.get('lightgbm', False)
    
    if sklearn_available:
        print("[OK] RandomForest, LogisticRegression, GradientBoosting 利用可能")
    else:
        print("[NG] 基本機械学習モデル 利用不可")
    
    if xgboost_available:
        print("[OK] XGBoost 利用可能")
    else:
        print("[NG] XGBoost 利用不可")
    
    if lightgbm_available:
        print("[OK] LightGBM 利用可能")
    else:
        print("[NG] LightGBM 利用不可")
    
    if test_result:
        print("[OK] システム動作確認完了")
    else:
        print("[注意] 一部機能に制限があります")
    
    print("\n次のステップ:")
    print("1. Streamlitダッシュボードを起動")
    print("2. 「取引実行」ボタンで予測開始")
    print("3. 「ML性能」タブで機械学習状況確認")
    print("4. 200以上の予測データが蓄積されると自動学習開始")
    
    if sklearn_available:
        print("\n期待される改善:")
        print("- 予測精度: 50% -> 60%+ を目標")
        print("- アンサンブル学習による安定性向上")
        print("- 高度特徴量による市場理解の深化")

if __name__ == "__main__":
    main()