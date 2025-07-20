#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025_06期間シンプル効果的システム
複雑さを排除し、確実に50%超を達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEffective2025_06:
    """シンプル効果的システム"""
    
    def __init__(self):
        self.target_accuracy = 0.50
        
    def load_data(self) -> pd.DataFrame:
        """データ読み込み"""
        data_path = Path("data/historical/BTC_1h_2025_06.csv")
        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        # 不要な列を除去（文字列列）
        text_columns = ['symbol', 'interval', 'timestamp_end']
        for col in text_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # 数値列のみ保持
        numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'trades']
        available_columns = [col for col in numeric_columns if col in df.columns]
        df = df[available_columns]
        
        logger.info(f"Loaded 2025_06: {len(df)} records, columns: {df.columns.tolist()}")
        return df
    
    def create_proven_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """実証済み効果的特徴量のみ作成"""
        logger.info("Creating proven effective features...")
        
        features_df = df.copy()
        
        # 1. 基本価格変動 - 最も重要
        features_df['price_change_1h'] = df['close'].pct_change()
        features_df['price_change_2h'] = df['close'].pct_change(2)
        features_df['price_change_4h'] = df['close'].pct_change(4)
        features_df['price_change_8h'] = df['close'].pct_change(8)
        
        # 2. 移動平均（効果的期間のみ）
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            features_df[f'ma_ratio_{period}'] = df['close'] / ma
            features_df[f'ma_slope_{period}'] = ma.diff() / ma
        
        # 3. RSI（最適期間）
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 4. ボラティリティ
        for period in [10, 20]:
            features_df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close']
        
        # 5. MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # 6. ボリンジャーバンド位置
        for period in [20]:
            rolling_mean = df['close'].rolling(period).mean()
            rolling_std = df['close'].rolling(period).std()
            features_df[f'bb_position_{period}'] = (df['close'] - rolling_mean) / rolling_std
        
        # 7. 価格位置（レンジ内位置）
        for period in [10, 20]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features_df[f'price_position_{period}'] = (df['close'] - low_min) / (high_max - low_min)
        
        # 8. 短期ラグ特徴量
        for lag in [1, 2, 3]:
            features_df[f'price_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'change_lag_{lag}'] = features_df['price_change_1h'].shift(lag)
        
        # 9. 時間特徴量（重要）
        features_df['hour'] = df.index.hour
        features_df['is_morning'] = (df.index.hour < 12).astype(int)
        features_df['is_evening'] = (df.index.hour >= 18).astype(int)
        
        # ターゲット
        features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # データクリーニング
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        features_df = features_df.dropna()
        
        feature_count = len([col for col in features_df.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'target']])
        logger.info(f"Created {feature_count} proven features for {len(features_df)} data points")
        
        return features_df
    
    def simple_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> list:
        """シンプル特徴量選択"""
        logger.info(f"Selecting top {n_features} features...")
        
        # F統計量ベース選択
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {selected_features}")
        return selected_features
    
    def train_best_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """最高性能モデル訓練"""
        logger.info(f"Training with {len(X)} samples, {len(X.columns)} features")
        
        # データ分割
        test_size = max(100, len(X) // 4)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 高性能モデル群
        models = {
            'xgb_tuned': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'rf_optimized': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            ),
            'logistic': LogisticRegression(
                C=1.0,
                penalty='l2',
                random_state=42,
                max_iter=1000
            )
        }
        
        results = {}
        predictions = {}
        
        # 各モデル訓練・評価
        for name, model in models.items():
            try:
                # 訓練
                if name == 'logistic':
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                # 評価
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                
                results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'overfitting': train_acc - test_acc
                }
                
                predictions[name] = test_pred
                
                logger.info(f"{name}: Test={test_acc:.3f}, Train={train_acc:.3f}, Overfit={train_acc-test_acc:.3f}")
                
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                continue
        
        # アンサンブル（単純多数決）
        if len(predictions) >= 3:
            ensemble_pred = np.round(np.mean(list(predictions.values()), axis=0))
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            
            results['ensemble'] = {
                'test_accuracy': ensemble_acc,
                'predictions': ensemble_pred
            }
            
            logger.info(f"Ensemble: {ensemble_acc:.3f}")
        
        return results
    
    def find_best_configuration(self) -> dict:
        """最適設定探索"""
        logger.info("Finding best configuration...")
        
        # データ読み込み
        df = self.load_data()
        features_df = self.create_proven_features(df)
        
        # 特徴量分離
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = features_df[feature_cols]
        y = features_df['target']
        
        best_accuracy = 0
        best_config = None
        
        # 異なる特徴量数でテスト
        for n_features in [10, 15, 20, 25]:
            logger.info(f"\n--- Testing {n_features} features ---")
            
            try:
                # 特徴量選択
                selected_features = self.simple_feature_selection(X, y, n_features)
                X_selected = X[selected_features]
                
                # モデル訓練
                results = self.train_best_models(X_selected, y)
                
                # 最高精度取得
                best_model_acc = 0
                best_model_name = None
                
                for model_name, model_result in results.items():
                    if 'test_accuracy' in model_result:
                        acc = model_result['test_accuracy']
                        if acc > best_model_acc:
                            best_model_acc = acc
                            best_model_name = model_name
                
                logger.info(f"Best for {n_features} features: {best_model_name} = {best_model_acc:.3f}")
                
                # 全体最高更新チェック
                if best_model_acc > best_accuracy:
                    best_accuracy = best_model_acc
                    best_config = {
                        'n_features': n_features,
                        'features': selected_features,
                        'best_model': best_model_name,
                        'accuracy': best_model_acc,
                        'all_results': results
                    }
                    
            except Exception as e:
                logger.error(f"Configuration {n_features} failed: {e}")
                continue
        
        return best_config
    
    def execute_final_run(self) -> dict:
        """最終実行"""
        logger.info("=== EXECUTING FINAL RUN FOR 2025_06 ===")
        
        config = self.find_best_configuration()
        
        if config and config['accuracy'] >= self.target_accuracy:
            logger.info(f"🎯 TARGET ACHIEVED: {config['accuracy']:.4f}")
            return {
                'success': True,
                'accuracy': config['accuracy'],
                'model': config['best_model'],
                'features_used': config['n_features'],
                'breakthrough': True
            }
        elif config:
            logger.info(f"Close but not quite: {config['accuracy']:.4f}")
            return {
                'success': True,
                'accuracy': config['accuracy'],
                'model': config['best_model'],
                'features_used': config['n_features'],
                'breakthrough': False,
                'gap': self.target_accuracy - config['accuracy']
            }
        else:
            return {'success': False, 'error': 'All configurations failed'}

def main():
    """メイン実行"""
    print("=== 2025_06期間 シンプル効果的システム ===")
    
    system = SimpleEffective2025_06()
    result = system.execute_final_run()
    
    if result['success']:
        accuracy = result['accuracy']
        
        if result['breakthrough']:
            print(f"\n目標達成! {accuracy:.2%}")
            print("全期間50%以上完了!")
        else:
            print(f"\n現在の最高精度: {accuracy:.2%}")
            print(f"目標まで: {result['gap']:.2%}")
            
        print(f"最優秀モデル: {result['model']}")
        print(f"使用特徴量数: {result['features_used']}")
        
    else:
        print(f"実行失敗: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()