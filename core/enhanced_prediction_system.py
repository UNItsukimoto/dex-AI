#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
強化予測システム
オーバーフィッティング対策と特徴量選択改良版
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LassoCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy import stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictionSystem:
    """強化予測システム - オーバーフィッティング対策版"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_features = []
        
    def load_data(self, period: str) -> pd.DataFrame:
        """データ読み込み"""
        try:
            data_path = Path(f"data/historical/BTC_1h_{period}.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            logger.info(f"Loaded {period}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {period}: {e}")
            return pd.DataFrame()
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """強化特徴量作成 - データ適応型"""
        if len(df) < 100:
            logger.warning(f"Insufficient data: {len(df)} rows")
            return df
        
        data_len = len(df)
        max_period = min(30, data_len // 10)  # より保守的な期間設定
        
        logger.info(f"Creating enhanced features for {data_len} data points...")
        
        features_df = df.copy()
        
        # 基本的な価格変動
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_change_abs'] = features_df['price_change'].abs()
        
        # 適応的移動平均
        for period in [5, 10, 20]:
            if period <= max_period:
                features_df[f'ma_{period}'] = df['close'].rolling(period).mean()
                features_df[f'ma_ratio_{period}'] = df['close'] / features_df[f'ma_{period}']
        
        # ボラティリティ指標
        for period in [5, 10, 15]:
            if period <= max_period:
                features_df[f'volatility_{period}'] = df['close'].rolling(period).std()
                features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / df['close']
        
        # RSI (Relative Strength Index)
        for period in [7, 14]:
            if period <= max_period:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 出来高指標
        if 'volume' in df.columns:
            for period in [5, 10]:
                if period <= max_period:
                    features_df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
                    features_df[f'volume_ratio_{period}'] = df['volume'] / features_df[f'volume_ma_{period}']
        
        # ボリンジャーバンド
        for period in [10, 20]:
            if period <= max_period:
                rolling_mean = df['close'].rolling(period).mean()
                rolling_std = df['close'].rolling(period).std()
                features_df[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
                features_df[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
                features_df[f'bb_ratio_{period}'] = (df['close'] - rolling_mean) / rolling_std
        
        # MACD
        if max_period >= 12:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Stochastic Oscillator
        for period in [10, 14]:
            if period <= max_period:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                features_df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                features_df[f'stoch_d_{period}'] = features_df[f'stoch_k_{period}'].rolling(3).mean()
        
        # ATR (Average True Range)
        for period in [10, 14]:
            if period <= max_period:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                features_df[f'atr_{period}'] = true_range.rolling(period).mean()
                features_df[f'atr_ratio_{period}'] = features_df[f'atr_{period}'] / df['close']
        
        # 時系列ラグ特徴量
        for lag in [1, 2, 3, 5]:
            if lag < data_len // 20:
                features_df[f'price_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'change_lag_{lag}'] = features_df['price_change'].shift(lag)
        
        # ターゲット作成
        features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 無限値とNaNを除去
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        feature_count = len([col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']])
        logger.info(f"Created {feature_count} enhanced features for {len(features_df)} data points")
        
        return features_df
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 40) -> List[str]:
        """改良特徴量選択 - 相関とMIベース"""
        logger.info("Starting enhanced feature selection...")
        
        # Step 1: 相関分析による事前フィルタリング
        target_correlations = {}
        for col in X.columns:
            try:
                corr, p_value = pearsonr(X[col].dropna(), y[X[col].dropna().index])
                if not np.isnan(corr) and p_value < 0.15:  # 基準緩和
                    target_correlations[col] = abs(corr)
            except:
                continue
        
        # 最低10特徴量は保持
        min_features = min(10, len(X.columns))
        if len(target_correlations) < min_features:
            # 相関が低くても上位特徴量を追加
            all_correlations = {}
            for col in X.columns:
                try:
                    corr, _ = pearsonr(X[col].dropna(), y[X[col].dropna().index])
                    if not np.isnan(corr):
                        all_correlations[col] = abs(corr)
                except:
                    continue
            
            top_all = sorted(all_correlations.items(), key=lambda x: x[1], reverse=True)[:min_features]
            target_correlations = dict(top_all)
        
        # 相関上位60%を選択（最低10特徴量）
        correlation_threshold = max(min_features, len(target_correlations) * 0.6)
        top_correlation_features = sorted(target_correlations.items(), 
                                        key=lambda x: x[1], reverse=True)[:int(correlation_threshold)]
        correlation_features = [f[0] for f in top_correlation_features]
        
        logger.info(f"Correlation filtering: {len(correlation_features)} features selected")
        
        # Step 2: 相互情報量による選択
        try:
            X_filtered = X[correlation_features]
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(max_features, len(correlation_features)))
            mi_selector.fit(X_filtered, y)
            mi_features = X_filtered.columns[mi_selector.get_support()].tolist()
            logger.info(f"Mutual information: {len(mi_features)} features selected")
        except Exception as e:
            logger.warning(f"MI selection failed: {e}, using correlation features")
            mi_features = correlation_features[:max_features]
        
        # Step 3: Random Forest による重要度チェック
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            rf.fit(X[mi_features], y)
            feature_importance = pd.Series(rf.feature_importances_, index=mi_features)
            
            # 重要度上位features選択
            top_importance_features = feature_importance.nlargest(max_features).index.tolist()
            logger.info(f"Random Forest importance: {len(top_importance_features)} features selected")
            
            return top_importance_features
            
        except Exception as e:
            logger.warning(f"RF importance failed: {e}, using MI features")
            return mi_features[:max_features]
    
    def create_robust_models(self) -> Dict:
        """正則化強化モデル群作成"""
        models = {
            'ridge': Ridge(alpha=10.0, random_state=42),  # 強い正則化
            'lasso': LassoCV(cv=3, random_state=42, max_iter=2000),
            'rf': RandomForestRegressor(
                n_estimators=100, 
                max_depth=8,  # 制限
                min_samples_split=20,  # 増加
                min_samples_leaf=10,   # 増加
                max_features='sqrt',
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,  # 制限
                learning_rate=0.05,  # 低下
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,  # L1正則化
                reg_lambda=2.0,  # L2正則化
                random_state=42
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,  # 制限
                learning_rate=0.05,  # 低下
                subsample=0.8,
                random_state=42
            )
        }
        return models
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, period: str) -> Dict:
        """時系列クロスバリデーション訓練"""
        logger.info(f"Training models for {period} with {len(X)} samples, {len(X.columns)} features")
        
        # データ分割
        test_size = max(100, len(X) // 5)  # 最低100サンプル
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # スケーリング（Robust Scalerでアウトライア対策）
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # モデル訓練と評価
        models = self.create_robust_models()
        results = {'models': {}, 'predictions': {}, 'scores': {}}
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in models.items():
            try:
                # クロスバリデーション
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error')
                
                # フルデータで訓練
                model.fit(X_train_scaled, y_train)
                
                # 予測
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # 分類用の変換
                train_pred_binary = (train_pred > 0.5).astype(int)
                test_pred_binary = (test_pred > 0.5).astype(int)
                y_train_binary = (y_train > 0.5).astype(int)
                y_test_binary = (y_test > 0.5).astype(int)
                
                # スコア計算
                train_acc = accuracy_score(y_train_binary, train_pred_binary)
                test_acc = accuracy_score(y_test_binary, test_pred_binary)
                
                results['models'][name] = model
                results['predictions'][name] = test_pred_binary
                results['scores'][name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_score': np.mean(cv_scores),
                    'overfitting': train_acc - test_acc
                }
                
                logger.info(f"{name}: Test={test_acc:.3f}, Train={train_acc:.3f}, Overfit={train_acc-test_acc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # アンサンブル予測（重み付き）
        if results['predictions']:
            ensemble_weights = {}
            total_weight = 0
            
            for name, scores in results['scores'].items():
                # オーバーフィッティングペナルティ付き重み
                weight = scores['test_accuracy'] * (1 - scores['overfitting'])
                weight = max(0.1, weight)  # 最小重み
                ensemble_weights[name] = weight
                total_weight += weight
            
            # 正規化
            for name in ensemble_weights:
                ensemble_weights[name] /= total_weight
            
            # アンサンブル予測
            ensemble_pred = np.zeros(len(X_test))
            for name, pred in results['predictions'].items():
                ensemble_pred += pred * ensemble_weights[name]
            
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            ensemble_acc = accuracy_score(y_test_binary, ensemble_pred_binary)
            
            results['ensemble'] = {
                'predictions': ensemble_pred_binary,
                'accuracy': ensemble_acc,
                'weights': ensemble_weights
            }
            
            logger.info(f"Ensemble accuracy: {ensemble_acc:.3f}")
        
        # スケーラー保存
        self.scalers[period] = scaler
        
        return results
    
    def predict_period(self, period: str) -> Dict:
        """期間別予測実行"""
        logger.info(f"\n=== {period} 期間の分析 ===")
        
        # データ読み込み
        df = self.load_data(period)
        if df.empty:
            return {'error': f'No data for period {period}'}
        
        logger.info(f"{period} の分析開始 - データ数: {len(df)}")
        
        # 特徴量作成
        features_df = self.create_enhanced_features(df)
        if len(features_df) < 50:
            return {'error': f'Insufficient processed data: {len(features_df)}'}
        
        # 特徴量とターゲット分離
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = features_df[feature_cols]
        y = features_df['target']
        
        # 特徴量選択
        selected_features = self.select_best_features(X, y, max_features=35)
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")
        
        # モデル訓練
        results = self.train_with_cv(X_selected, y, period)
        
        # 結果まとめ
        if 'ensemble' in results:
            final_accuracy = results['ensemble']['accuracy']
            logger.info(f"{period} 最終精度: {final_accuracy:.2%}")
            
            return {
                'period': period,
                'accuracy': final_accuracy,
                'data_points': len(features_df),
                'features_used': len(selected_features),
                'model_scores': results['scores'],
                'ensemble_weights': results['ensemble']['weights']
            }
        else:
            return {'error': 'Training failed for all models'}

def main():
    """メイン実行"""
    print("強化予測システム開始...")
    
    system = EnhancedPredictionSystem()
    target_periods = ['2025_06', 'current', '2025_05']
    results = {}
    
    for period in target_periods:
        try:
            result = system.predict_period(period)
            results[period] = result
            
            if 'accuracy' in result:
                print(f"{period}: {result['accuracy']:.2%}")
            else:
                print(f"{period}: エラー - {result.get('error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"Period {period} failed: {e}")
            results[period] = {'error': str(e)}
    
    # 結果サマリー
    print("\n=== 強化システム結果 ===")
    total_improved = 0
    for period, result in results.items():
        if 'accuracy' in result:
            acc = result['accuracy']
            improved = "OK" if acc >= 0.50 else "NG"
            print(f"{period}: {acc:.2%} {improved}")
            if acc >= 0.50:
                total_improved += 1
        else:
            print(f"{period}: エラー")
    
    print(f"\n50%達成: {total_improved}/{len(target_periods)} 期間")

if __name__ == "__main__":
    main()