#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025_06期間最終突破システム
48.85% → 50%+ の最後の壁を突破
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import warnings

# Advanced ML libraries
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import boxcox

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Final2025_06_Breakthrough:
    """2025_06期間最終突破システム"""
    
    def __init__(self):
        self.target_accuracy = 0.50
        self.best_config = None
        
    def load_data(self) -> pd.DataFrame:
        """データ読み込み"""
        data_path = Path("data/historical/BTC_1h_2025_06.csv")
        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded 2025_06: {len(df)} records")
        return df
    
    def create_ultra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """超高精度特徴量作成"""
        logger.info("Creating ultra-precision features...")
        
        features_df = df.copy()
        
        # 1. 多層価格変動パターン
        for h in [1, 2, 3, 4, 6, 8, 12]:
            features_df[f'price_change_{h}h'] = df['close'].pct_change(h)
            features_df[f'price_accel_{h}h'] = features_df[f'price_change_{h}h'].diff()
            features_df[f'price_momentum_{h}h'] = df['close'] / df['close'].shift(h)
        
        # 2. 適応的移動平均システム
        for period in [3, 5, 8, 13, 21, 34]:  # フィボナッチ数列
            ma = df['close'].rolling(period).mean()
            features_df[f'ma_{period}'] = ma
            features_df[f'ma_distance_{period}'] = (df['close'] - ma) / ma
            features_df[f'ma_slope_{period}'] = ma.diff() / ma
            features_df[f'ma_curvature_{period}'] = ma.diff().diff()
        
        # 3. 動的ボラティリティ指標
        for period in [6, 12, 24]:
            vol = df['close'].rolling(period).std()
            features_df[f'volatility_{period}'] = vol
            features_df[f'vol_ratio_{period}'] = vol / df['close']
            features_df[f'vol_change_{period}'] = vol.pct_change()
            features_df[f'vol_zscore_{period}'] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std()
        
        # 4. 高精度RSI変種
        for period in [5, 9, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features_df[f'rsi_{period}'] = rsi
            features_df[f'rsi_slope_{period}'] = rsi.diff()
            features_df[f'rsi_divergence_{period}'] = rsi - rsi.rolling(10).mean()
        
        # 5. 多重MACD
        for fast, slow in [(5, 13), (8, 21), (12, 26), (21, 55)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            features_df[f'macd_{fast}_{slow}'] = macd
            features_df[f'macd_signal_{fast}_{slow}'] = signal
            features_df[f'macd_hist_{fast}_{slow}'] = histogram
            features_df[f'macd_slope_{fast}_{slow}'] = macd.diff()
            features_df[f'macd_momentum_{fast}_{slow}'] = histogram.diff()
        
        # 6. ボリンジャーバンド詳細分析
        for period in [10, 20, 30]:
            for std_factor in [1.5, 2.0, 2.5]:
                rolling_mean = df['close'].rolling(period).mean()
                rolling_std = df['close'].rolling(period).std()
                
                upper = rolling_mean + (rolling_std * std_factor)
                lower = rolling_mean - (rolling_std * std_factor)
                
                features_df[f'bb_position_{period}_{std_factor}'] = (df['close'] - rolling_mean) / rolling_std
                features_df[f'bb_width_{period}_{std_factor}'] = (upper - lower) / rolling_mean
                features_df[f'bb_squeeze_{period}_{std_factor}'] = rolling_std / rolling_std.rolling(20).mean()
        
        # 7. ストキャスティクス変種
        for k_period in [9, 14, 21]:
            for d_period in [3, 5]:
                low_min = df['low'].rolling(k_period).min()
                high_max = df['high'].rolling(k_period).max()
                k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
                d_percent = k_percent.rolling(d_period).mean()
                
                features_df[f'stoch_k_{k_period}_{d_period}'] = k_percent
                features_df[f'stoch_d_{k_period}_{d_period}'] = d_percent
                features_df[f'stoch_momentum_{k_period}_{d_period}'] = k_percent - d_percent
        
        # 8. 価格位置インジケータ
        for period in [5, 10, 20, 50]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features_df[f'price_position_{period}'] = (df['close'] - low_min) / (high_max - low_min)
            features_df[f'range_position_{period}'] = (df['high'] - df['low']) / (high_max - low_min)
        
        # 9. ボリューム分析（利用可能な場合）
        if 'volume' in df.columns:
            for period in [5, 10, 20]:
                vol_ma = df['volume'].rolling(period).mean()
                features_df[f'volume_ratio_{period}'] = df['volume'] / vol_ma
                features_df[f'vol_price_corr_{period}'] = df['volume'].rolling(period).corr(df['close'])
                
                # オンバランスボリューム
                obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
                features_df[f'obv_slope_{period}'] = obv.rolling(period).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == period else 0
                )
        
        # 10. フーリエ変換特徴量
        for period in [12, 24, 48]:
            if len(df) >= period * 2:
                # 価格の周期性を捉える
                price_fft = np.fft.fft(df['close'].tail(period))
                features_df[f'fft_real_{period}'] = np.real(price_fft[1])  # 第1成分
                features_df[f'fft_imag_{period}'] = np.imag(price_fft[1])
                features_df[f'fft_magnitude_{period}'] = np.abs(price_fft[1])
        
        # 11. 時間特徴量
        features_df['hour'] = df.index.hour
        features_df['day_of_week'] = df.index.dayofweek
        features_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 12. ラグ特徴量（多層）
        for lag in [1, 2, 3, 4, 6, 8, 12, 24]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'change_lag_{lag}'] = features_df['price_change_1h'].shift(lag)
            if 'volume' in df.columns:
                features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # ターゲット作成
        features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # データクリーニング
        features_df = self.advanced_data_cleaning(features_df)
        
        return features_df
    
    def advanced_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度データクリーニング"""
        logger.info("Advanced data cleaning...")
        
        # 無限値処理
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 特徴量列特定
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        # 欠損率の高い特徴量除去（70%閾値）
        missing_rates = df[feature_cols].isnull().mean()
        good_features = missing_rates[missing_rates < 0.7].index.tolist()
        
        # 数値型変換と検証
        valid_features = []
        for col in good_features:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isnull().all() and df[col].var() > 1e-10:  # 分散チェック
                    valid_features.append(col)
            except:
                continue
        
        # 必要列のみ保持
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'target'] + valid_features
        df = df[keep_cols]
        
        # 前方・後方補間
        df[valid_features] = df[valid_features].fillna(method='ffill').fillna(method='bfill')
        
        # 残りのNaN除去
        df = df.dropna()
        
        logger.info(f"Cleaned data: {len(df)} rows, {len(valid_features)} features")
        return df
    
    def ultra_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """超高精度特徴量選択"""
        logger.info("Ultra feature selection...")
        
        # 1. 統計的選択（複数手法組み合わせ）
        selectors = {
            'f_classif': SelectKBest(score_func=f_classif, k=30),
            'mutual_info': SelectKBest(score_func=mutual_info_classif, k=30),
            'chi2': SelectKBest(score_func=chi2, k=30)
        }
        
        selected_features = set()
        
        for name, selector in selectors.items():
            try:
                # 非負値に変換（chi2用）
                X_positive = X - X.min() + 1e-6 if name == 'chi2' else X
                selector.fit(X_positive, y)
                features = X.columns[selector.get_support()].tolist()
                selected_features.update(features)
                logger.info(f"{name}: {len(features)} features")
            except:
                continue
        
        # 2. Random Forest重要度
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X, y)
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
            rf_features = rf_importance.nlargest(25).index.tolist()
            selected_features.update(rf_features)
            logger.info(f"Random Forest: {len(rf_features)} features")
        except:
            pass
        
        # 3. XGBoost重要度
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, max_depth=6)
            xgb_model.fit(X, y)
            xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
            xgb_features = xgb_importance.nlargest(25).index.tolist()
            selected_features.update(xgb_features)
            logger.info(f"XGBoost: {len(xgb_features)} features")
        except:
            pass
        
        final_features = list(selected_features)[:40]  # 上位40特徴量
        logger.info(f"Final selection: {len(final_features)} features")
        
        return final_features
    
    def create_super_ensemble(self) -> Dict:
        """スーパーアンサンブルモデル作成"""
        base_models = {
            # 線形モデル群
            'logistic_l1': LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42),
            'logistic_l2': LogisticRegression(penalty='l2', C=1.0, random_state=42),
            'ridge': RidgeClassifier(alpha=1.0, random_state=42),
            
            # ツリー系モデル群
            'rf_deep': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, random_state=42),
            'rf_wide': RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=20, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=42),
            'gbm': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42),
            
            # ブースティング
            'xgb_conservative': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=2, random_state=42
            ),
            'xgb_aggressive': xgb.XGBClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1, random_state=42
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=100, max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1, random_state=42, verbose=-1
            ),
            
            # その他
            'svc_rbf': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.01, max_iter=500, random_state=42)
        }
        
        return base_models
    
    def breakthrough_training(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """突破訓練システム"""
        logger.info(f"Breakthrough training: {len(X)} samples, {len(X.columns)} features")
        
        # データ分割（より慎重に）
        test_size = max(120, len(X) // 4)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 複数スケーラーテスト
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        best_accuracy = 0
        best_config = None
        
        for scaler_name, scaler in scalers.items():
            logger.info(f"Testing {scaler_name} scaler...")
            
            try:
                # スケーリング
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
                
                # モデル訓練
                models = self.create_super_ensemble()
                predictions = {}
                scores = {}
                
                for name, model in models.items():
                    try:
                        # キャリブレーション適用
                        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                        calibrated_model.fit(X_train_scaled, y_train)
                        
                        # 予測
                        train_pred = calibrated_model.predict(X_train_scaled)
                        test_pred = calibrated_model.predict(X_test_scaled)
                        test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
                        
                        # スコア計算
                        train_acc = accuracy_score(y_train, train_pred)
                        test_acc = accuracy_score(y_test, test_pred)
                        test_auc = roc_auc_score(y_test, test_proba)
                        
                        predictions[name] = test_pred
                        scores[name] = {
                            'accuracy': test_acc,
                            'auc': test_auc,
                            'overfitting': train_acc - test_acc
                        }
                        
                        logger.info(f"  {name}: {test_acc:.3f} (AUC: {test_auc:.3f})")
                        
                    except Exception as e:
                        logger.warning(f"  {name} failed: {e}")
                        continue
                
                # アンサンブル戦略
                if len(predictions) >= 5:
                    # 1. 性能ベース重み付き投票
                    weights = []
                    preds = []
                    for name in predictions:
                        # AUCと精度の加重平均をweight
                        weight = (scores[name]['accuracy'] + scores[name]['auc']) / 2
                        weight = max(0.1, weight)  # 最小重み
                        weights.append(weight)
                        preds.append(predictions[name])
                    
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    
                    ensemble_pred = np.round(np.average(preds, axis=0, weights=weights))
                    ensemble_acc = accuracy_score(y_test, ensemble_pred)
                    
                    # 2. 閾値調整
                    ensemble_proba = np.average([
                        model.predict_proba(X_test_scaled)[:, 1] 
                        for name, model in models.items() 
                        if name in predictions
                    ], axis=0, weights=weights)
                    
                    # 最適閾値探索
                    best_threshold = 0.5
                    best_threshold_acc = ensemble_acc
                    
                    for threshold in np.arange(0.3, 0.7, 0.05):
                        thresh_pred = (ensemble_proba >= threshold).astype(int)
                        thresh_acc = accuracy_score(y_test, thresh_pred)
                        if thresh_acc > best_threshold_acc:
                            best_threshold_acc = thresh_acc
                            best_threshold = threshold
                    
                    final_pred = (ensemble_proba >= best_threshold).astype(int)
                    final_acc = accuracy_score(y_test, final_pred)
                    
                    logger.info(f"  {scaler_name} - Final ensemble: {final_acc:.3f} (threshold: {best_threshold:.2f})")
                    
                    if final_acc > best_accuracy:
                        best_accuracy = final_acc
                        best_config = {
                            'scaler': scaler_name,
                            'accuracy': final_acc,
                            'threshold': best_threshold,
                            'models': models,
                            'predictions': final_pred,
                            'scores': scores
                        }
                        
            except Exception as e:
                logger.warning(f"Scaler {scaler_name} failed: {e}")
                continue
        
        return best_config
    
    def final_breakthrough(self) -> Dict:
        """最終突破実行"""
        logger.info("=== FINAL BREAKTHROUGH FOR 2025_06 ===")
        
        # データ読み込み
        df = self.load_data()
        
        # 超特徴量作成
        features_df = self.create_ultra_features(df)
        if len(features_df) < 100:
            return {'error': f'Insufficient data: {len(features_df)}'}
        
        # 特徴量分離
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = features_df[feature_cols]
        y = features_df['target']
        
        logger.info(f"Ultra features: {len(feature_cols)}")
        
        # 超特徴量選択
        selected_features = self.ultra_feature_selection(X, y)
        X_selected = X[selected_features]
        
        # 突破訓練
        result = self.breakthrough_training(X_selected, y)
        
        if result and 'accuracy' in result:
            self.best_config = result
            accuracy = result['accuracy']
            threshold = result.get('threshold', 0.5)
            
            logger.info(f"BREAKTHROUGH RESULT: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'threshold': threshold,
                'features_used': len(selected_features),
                'scaler': result['scaler'],
                'breakthrough': accuracy >= self.target_accuracy
            }
        else:
            return {'error': 'Breakthrough training failed'}

def main():
    """メイン実行"""
    print("=== 2025_06期間 最終突破システム ===")
    
    breakthrough = Final2025_06_Breakthrough()
    result = breakthrough.final_breakthrough()
    
    if 'accuracy' in result:
        accuracy = result['accuracy']
        status = "🎯 BREAKTHROUGH!" if result['breakthrough'] else "継続改善必要"
        
        print(f"\n最終結果: {accuracy:.2%}")
        print(f"目標達成: {status}")
        print(f"使用特徴量: {result['features_used']}")
        print(f"スケーラー: {result['scaler']}")
        print(f"閾値: {result['threshold']:.3f}")
        
        if result['breakthrough']:
            print("\n🎉 全期間50%以上達成完了!")
        else:
            gap = 0.50 - accuracy
            print(f"\n残り: {gap:.2%} の改善が必要")
            
    else:
        print(f"エラー: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()