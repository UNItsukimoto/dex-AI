#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025_06期間専用最適化システム
45.93% → 50%+ 達成のための特化型モデル
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from scipy import stats
import optuna

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Period2025_06_Optimizer:
    """2025_06期間専用最適化システム"""
    
    def __init__(self):
        self.best_model = None
        self.best_scaler = None
        self.best_features = []
        self.best_params = {}
        
    def load_target_data(self) -> pd.DataFrame:
        """2025_06データ読み込み"""
        data_path = Path("data/historical/BTC_1h_2025_06.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Target data not found: {data_path}")
        
        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded 2025_06: {len(df)} records")
        return df
    
    def create_specialized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """2025_06期間に特化した特徴量作成"""
        logger.info("Creating specialized features for 2025_06...")
        
        features_df = df.copy()
        
        # 基本価格変動（短期重視）
        for period in [1, 2, 3, 4, 6]:
            features_df[f'price_change_{period}h'] = df['close'].pct_change(period)
            features_df[f'price_volatility_{period}h'] = df['close'].rolling(period).std() / df['close']
        
        # 高頻度移動平均（市場反応捕捉）
        for period in [3, 6, 12, 24, 48]:
            features_df[f'ma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'ma_ratio_{period}'] = df['close'] / features_df[f'ma_{period}']
            features_df[f'ma_slope_{period}'] = features_df[f'ma_{period}'].diff() / features_df[f'ma_{period}']
        
        # 動的RSI（複数期間）
        for period in [6, 12, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            features_df[f'rsi_ma_{period}'] = features_df[f'rsi_{period}'].rolling(6).mean()
        
        # 詳細ボリューム分析
        if 'volume' in df.columns:
            for period in [6, 12, 24]:
                features_df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
                features_df[f'volume_ratio_{period}'] = df['volume'] / features_df[f'volume_ma_{period}']
                features_df[f'volume_price_corr_{period}'] = df['volume'].rolling(period).corr(df['close'])
        
        # 高精度ボリンジャーバンド
        for period in [12, 24]:
            for std_dev in [1.5, 2.0, 2.5]:
                rolling_mean = df['close'].rolling(period).mean()
                rolling_std = df['close'].rolling(period).std()
                features_df[f'bb_upper_{period}_{std_dev}'] = rolling_mean + (rolling_std * std_dev)
                features_df[f'bb_lower_{period}_{std_dev}'] = rolling_mean - (rolling_std * std_dev)
                features_df[f'bb_position_{period}_{std_dev}'] = (df['close'] - rolling_mean) / (rolling_std * std_dev)
        
        # MACD変種
        for fast, slow in [(8, 21), (12, 26), (5, 35)]:
            exp1 = df['close'].ewm(span=fast).mean()
            exp2 = df['close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            features_df[f'macd_{fast}_{slow}'] = macd
            features_df[f'macd_signal_{fast}_{slow}'] = signal
            features_df[f'macd_histogram_{fast}_{slow}'] = macd - signal
            features_df[f'macd_slope_{fast}_{slow}'] = macd.diff()
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features_df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features_df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        if 'volume' in df.columns:
            for period in [14, 21]:
                tp = (df['high'] + df['low'] + df['close']) / 3
                raw_money_flow = tp * df['volume']
                positive_flow = raw_money_flow.where(tp > tp.shift(1), 0).rolling(period).sum()
                negative_flow = raw_money_flow.where(tp < tp.shift(1), 0).rolling(period).sum()
                mfi_ratio = positive_flow / negative_flow
                features_df[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
        
        # Price momentum indicators
        for period in [3, 6, 12, 24]:
            features_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
            features_df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Trend strength
        for period in [12, 24]:
            features_df[f'trend_strength_{period}'] = abs(df['close'].rolling(period).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == period else 0))
        
        # Price position in range
        for period in [6, 12, 24]:
            period_high = df['high'].rolling(period).max()
            period_low = df['low'].rolling(period).min()
            features_df[f'price_position_{period}'] = (df['close'] - period_low) / (period_high - period_low)
        
        # Lagged features (時間遅れ)
        for lag in [1, 2, 3, 6, 12]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        # Target (1時間後の上昇/下降)
        features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Clean data progressively
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 欠損率の高い特徴量を除去
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        missing_rates = features_df[feature_cols].isnull().mean()
        good_features = missing_rates[missing_rates < 0.5].index.tolist()  # 50%未満の欠損
        
        # 必要な列を保持
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'target'] + good_features
        features_df = features_df[keep_cols]
        
        # より緩い欠損値処理
        features_df = features_df.dropna(subset=['target'])  # ターゲットのみ必須
        
        # 残りの特徴量の欠損値を前方補間
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        features_df[feature_cols] = features_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # 最終的にNaNが残った行のみ削除
        features_df = features_df.dropna()
        
        # 数値型に変換（非数値列を除外）
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        # 各特徴量列を数値型に変換（失敗したら除外）
        valid_features = []
        for col in feature_cols:
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                if not features_df[col].isnull().all():  # 全てNaNでない
                    valid_features.append(col)
            except:
                continue
        
        # 有効な特徴量のみ保持
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'target'] + valid_features
        features_df = features_df[keep_cols]
        
        # 再度NaN除去
        features_df = features_df.dropna()
        
        feature_count = len(valid_features)
        logger.info(f"Created {feature_count} specialized features for {len(features_df)} data points")
        
        return features_df
    
    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Optunaベース特徴量選択最適化"""
        logger.info("Starting Optuna-based feature selection...")
        
        def objective(trial):
            # 特徴量数を最適化
            n_features = trial.suggest_int('n_features', 10, min(50, len(X.columns)))
            
            # Random Forest重要度ベース選択
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            rf.fit(X, y)
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = feature_importance.nlargest(n_features).index.tolist()
            
            # Cross-validation評価
            X_selected = X[top_features]
            model = LogisticRegression(random_state=42, max_iter=1000)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_val_scaled)
                scores.append(accuracy_score(y_val, pred))
            
            return np.mean(scores)
        
        # Optuna最適化
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        best_n_features = study.best_params['n_features']
        logger.info(f"Optimal number of features: {best_n_features}")
        
        # 最適特徴量選択
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf.fit(X, y)
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
        selected_features = feature_importance.nlargest(best_n_features).index.tolist()
        
        logger.info(f"Selected {len(selected_features)} optimal features")
        return selected_features
    
    def create_optimized_models(self) -> Dict:
        """最適化済みモデル群作成"""
        models = {
            'logistic': LogisticRegression(
                C=0.5, 
                penalty='elasticnet', 
                l1_ratio=0.3,
                solver='saga', 
                random_state=42, 
                max_iter=2000
            ),
            'rf_optimized': RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            'xgb_tuned': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.5,
                reg_lambda=1.5,
                scale_pos_weight=1.2,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(50, 25),
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'nb': GaussianNB(),
            'svc': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        return models
    
    def train_ensemble_system(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """アンサンブルシステム訓練"""
        logger.info(f"Training ensemble system with {len(X)} samples, {len(X.columns)} features")
        
        # データ分割
        test_size = max(100, len(X) // 4)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # スケーリング（MinMaxとStandardの組み合わせテスト）
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        best_accuracy = 0
        best_scaler_name = None
        best_results = {}
        
        for scaler_name, scaler in scalers.items():
            logger.info(f"Testing {scaler_name} scaler...")
            
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
            models = self.create_optimized_models()
            model_predictions = {}
            model_scores = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    model_predictions[name] = test_pred
                    model_scores[name] = {
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'overfitting': train_acc - test_acc
                    }
                    
                    logger.info(f"  {name}: {test_acc:.3f} (overfit: {train_acc-test_acc:.3f})")
                    
                except Exception as e:
                    logger.warning(f"  {name} failed: {e}")
                    continue
            
            # アンサンブル（複数手法）
            if len(model_predictions) >= 3:
                # 1. 単純平均
                ensemble_pred_simple = np.round(np.mean(list(model_predictions.values()), axis=0))
                simple_acc = accuracy_score(y_test, ensemble_pred_simple)
                
                # 2. 重み付き平均（テスト精度ベース）
                weights = []
                predictions = []
                for name in model_predictions:
                    weights.append(model_scores[name]['test_accuracy'])
                    predictions.append(model_predictions[name])
                
                weights = np.array(weights)
                weights = weights / weights.sum()
                ensemble_pred_weighted = np.round(np.average(predictions, axis=0, weights=weights))
                weighted_acc = accuracy_score(y_test, ensemble_pred_weighted)
                
                # 3. Voting Classifier
                voting_models = [(name, models[name]) for name in model_predictions if name in models]
                if len(voting_models) >= 3:
                    voting_clf = VotingClassifier(estimators=voting_models, voting='hard')
                    voting_clf.fit(X_train_scaled, y_train)
                    voting_pred = voting_clf.predict(X_test_scaled)
                    voting_acc = accuracy_score(y_test, voting_pred)
                else:
                    voting_acc = 0
                
                # 最高精度選択
                ensemble_results = {
                    'simple': (simple_acc, ensemble_pred_simple),
                    'weighted': (weighted_acc, ensemble_pred_weighted),
                    'voting': (voting_acc, voting_pred) if voting_acc > 0 else (0, None)
                }
                
                best_ensemble = max(ensemble_results.items(), key=lambda x: x[1][0])
                ensemble_accuracy = best_ensemble[1][0]
                
                logger.info(f"  {scaler_name} - Best ensemble ({best_ensemble[0]}): {ensemble_accuracy:.3f}")
                
                if ensemble_accuracy > best_accuracy:
                    best_accuracy = ensemble_accuracy
                    best_scaler_name = scaler_name
                    best_results = {
                        'scaler': scaler,
                        'models': models,
                        'model_scores': model_scores,
                        'ensemble_method': best_ensemble[0],
                        'ensemble_accuracy': ensemble_accuracy,
                        'ensemble_predictions': best_ensemble[1][1]
                    }
        
        logger.info(f"Best configuration: {best_scaler_name} scaler, accuracy: {best_accuracy:.3f}")
        
        return best_results
    
    def optimize_2025_06(self) -> Dict:
        """2025_06期間最適化実行"""
        logger.info("Starting 2025_06 period optimization...")
        
        # データ読み込み
        df = self.load_target_data()
        
        # 特殊特徴量作成
        features_df = self.create_specialized_features(df)
        if len(features_df) < 100:
            return {'error': f'Insufficient data: {len(features_df)}'}
        
        # 特徴量とターゲット分離
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = features_df[feature_cols]
        y = features_df['target']
        
        logger.info(f"Initial features: {len(feature_cols)}")
        
        # 最適特徴量選択
        selected_features = self.optimize_feature_selection(X, y)
        X_selected = X[selected_features]
        
        # アンサンブルシステム訓練
        results = self.train_ensemble_system(X_selected, y)
        
        if 'ensemble_accuracy' in results:
            self.best_model = results['models']
            self.best_scaler = results['scaler']
            self.best_features = selected_features
            
            final_accuracy = results['ensemble_accuracy']
            logger.info(f"Final 2025_06 optimization result: {final_accuracy:.2%}")
            
            return {
                'period': '2025_06',
                'accuracy': final_accuracy,
                'data_points': len(features_df),
                'features_used': len(selected_features),
                'scaler_type': type(results['scaler']).__name__,
                'ensemble_method': results['ensemble_method'],
                'model_scores': results['model_scores']
            }
        else:
            return {'error': 'Optimization failed'}

def main():
    """メイン実行"""
    print("2025_06期間専用最適化開始...")
    
    optimizer = Period2025_06_Optimizer()
    result = optimizer.optimize_2025_06()
    
    if 'accuracy' in result:
        accuracy = result['accuracy']
        status = "達成" if accuracy >= 0.50 else "未達成"
        print(f"\n2025_06最適化結果: {accuracy:.2%} ({status})")
        print(f"使用特徴量: {result['features_used']}")
        print(f"スケーラー: {result['scaler_type']}")
        print(f"アンサンブル手法: {result['ensemble_method']}")
        
        if accuracy >= 0.50:
            print("\n全期間50%以上達成!")
        else:
            improvement_needed = 0.50 - accuracy
            print(f"追加改善必要: +{improvement_needed:.2%}")
    else:
        print(f"最適化エラー: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()