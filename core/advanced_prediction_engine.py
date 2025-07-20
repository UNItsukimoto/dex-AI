#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高度予測エンジン
機械学習モデルとアンサンブル学習による予測精度向上システム
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import json
from pathlib import Path
import warnings

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("機械学習ライブラリが利用できません。基本予測モードで動作します。")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """高度特徴量エンジニアリング"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def create_technical_features(self, df: pd.DataFrame) -> Dict:
        """テクニカル特徴量作成"""
        if len(df) < 50:
            return {}
        
        features = {}
        
        try:
            # 基本価格特徴量
            features['price'] = df['close'].iloc[-1]
            features['returns_1h'] = df['close'].pct_change().iloc[-1]
            features['returns_4h'] = df['close'].pct_change(4).iloc[-1]
            features['returns_24h'] = df['close'].pct_change(24).iloc[-1] if len(df) >= 24 else 0
            
            # 移動平均系
            for period in [5, 10, 20, 50]:
                if len(df) >= period:
                    ma = df['close'].rolling(period).mean()
                    features[f'ma_{period}'] = ma.iloc[-1]
                    features[f'ma_ratio_{period}'] = df['close'].iloc[-1] / ma.iloc[-1]
                    features[f'ma_slope_{period}'] = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2] if len(ma) >= 2 else 0
            
            # ボラティリティ系
            for period in [10, 20, 30]:
                if len(df) >= period:
                    vol = df['close'].rolling(period).std()
                    features[f'volatility_{period}'] = vol.iloc[-1] / df['close'].iloc[-1]
                    features[f'volatility_ratio_{period}'] = vol.iloc[-1] / vol.mean()
            
            # RSI系
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1]
            features['rsi_slope'] = rsi.iloc[-1] - rsi.iloc[-2] if len(rsi) >= 2 else 0
            
            # MACD系
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = macd_signal.iloc[-1]
            features['macd_histogram'] = macd.iloc[-1] - macd_signal.iloc[-1]
            features['macd_slope'] = macd.iloc[-1] - macd.iloc[-2] if len(macd) >= 2 else 0
            
            # ボリンジャーバンド
            if len(df) >= 20:
                bb_ma = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                features['bb_upper'] = (bb_ma + 2 * bb_std).iloc[-1]
                features['bb_lower'] = (bb_ma - 2 * bb_std).iloc[-1]
                features['bb_position'] = (df['close'].iloc[-1] - bb_ma.iloc[-1]) / bb_std.iloc[-1]
                features['bb_width'] = (bb_std.iloc[-1] * 4) / bb_ma.iloc[-1]
            
            # 出来高特徴量
            if 'volume' in df.columns:
                features['volume'] = df['volume'].iloc[-1]
                features['volume_ma'] = df['volume'].rolling(20).mean().iloc[-1]
                features['volume_ratio'] = features['volume'] / features['volume_ma'] if features['volume_ma'] > 0 else 1
                features['volume_trend'] = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"テクニカル特徴量作成エラー: {e}")
            return {}
    
    def create_advanced_features(self, df: pd.DataFrame) -> Dict:
        """高度特徴量作成"""
        if len(df) < 100:
            return {}
        
        features = {}
        
        try:
            # フラクタル次元
            features['fractal_dimension'] = self.calculate_fractal_dimension(df['close'])
            
            # ハースト指数
            features['hurst_exponent'] = self.calculate_hurst_exponent(df['close'])
            
            # エントロピー
            features['entropy'] = self.calculate_entropy(df['close'])
            
            # サポート・レジスタンス強度
            support_resistance = self.find_support_resistance(df)
            features.update(support_resistance)
            
            # 市場構造変化検出
            regime_features = self.detect_regime_change(df)
            features.update(regime_features)
            
            # 相関特徴量（他の時間軸との相関）
            correlation_features = self.calculate_correlation_features(df)
            features.update(correlation_features)
            
            return features
            
        except Exception as e:
            logger.error(f"高度特徴量作成エラー: {e}")
            return {}
    
    def calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """フラクタル次元計算"""
        try:
            if len(prices) < 50:
                return 1.5
            
            # Higuchi方法による計算
            prices_array = prices.values
            N = len(prices_array)
            k_max = min(N // 4, 20)
            
            lk_values = []
            k_values = []
            
            for k in range(1, k_max + 1):
                lk = 0
                for m in range(k):
                    lm_k = 0
                    max_i = int((N - 1 - m) / k)
                    if max_i > 0:
                        for i in range(1, max_i + 1):
                            lm_k += abs(prices_array[m + i * k] - prices_array[m + (i - 1) * k])
                        lm_k = lm_k * (N - 1) / (max_i * k * k)
                        lk += lm_k
                
                if lk > 0:
                    lk_values.append(np.log(lk / k))
                    k_values.append(np.log(1 / k))
            
            if len(lk_values) > 1:
                slope = np.polyfit(k_values, lk_values, 1)[0]
                return max(1.0, min(2.0, slope))
            
            return 1.5
            
        except Exception as e:
            logger.error(f"フラクタル次元計算エラー: {e}")
            return 1.5
    
    def calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """ハースト指数計算"""
        try:
            if len(prices) < 100:
                return 0.5
            
            prices_array = prices.values
            N = len(prices_array)
            
            # R/S解析
            log_returns = np.diff(np.log(prices_array))
            mean_return = np.mean(log_returns)
            
            # 累積偏差
            deviations = np.cumsum(log_returns - mean_return)
            
            # レンジ
            R = np.max(deviations) - np.min(deviations)
            
            # 標準偏差
            S = np.std(log_returns)
            
            if S == 0:
                return 0.5
            
            # 複数の時間スケールで計算
            time_scales = [10, 20, 30, 50, 100]
            rs_values = []
            scale_values = []
            
            for scale in time_scales:
                if scale >= N:
                    continue
                
                chunks = N // scale
                rs_chunk = []
                
                for i in range(chunks):
                    start = i * scale
                    end = start + scale
                    chunk_returns = log_returns[start:end]
                    chunk_mean = np.mean(chunk_returns)
                    chunk_deviations = np.cumsum(chunk_returns - chunk_mean)
                    chunk_R = np.max(chunk_deviations) - np.min(chunk_deviations)
                    chunk_S = np.std(chunk_returns)
                    
                    if chunk_S > 0:
                        rs_chunk.append(chunk_R / chunk_S)
                
                if rs_chunk:
                    avg_rs = np.mean(rs_chunk)
                    if avg_rs > 0:
                        rs_values.append(np.log(avg_rs))
                        scale_values.append(np.log(scale))
            
            if len(rs_values) > 1:
                hurst = np.polyfit(scale_values, rs_values, 1)[0]
                return max(0.0, min(1.0, hurst))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"ハースト指数計算エラー: {e}")
            return 0.5
    
    def calculate_entropy(self, prices: pd.Series) -> float:
        """エントロピー計算"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 10:
                return 0.5
            
            # 収益率をビンに分割
            bins = 20
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist / len(returns)  # 正規化
            
            # エントロピー計算
            entropy = 0
            for p in hist:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # 正規化（最大エントロピーで割る）
            max_entropy = np.log2(bins)
            return entropy / max_entropy if max_entropy > 0 else 0.5
            
        except Exception as e:
            logger.error(f"エントロピー計算エラー: {e}")
            return 0.5
    
    def find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """サポート・レジスタンス検出"""
        try:
            features = {}
            
            if len(df) < 50:
                return features
            
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            current_price = closes[-1]
            
            # ローカル極値検出
            high_peaks = []
            low_peaks = []
            
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                    high_peaks.append((i, highs[i]))
                
                if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                    low_peaks.append((i, lows[i]))
            
            # レジスタンスレベル
            resistance_levels = [price for _, price in high_peaks if price > current_price]
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                features['resistance_distance'] = (nearest_resistance - current_price) / current_price
                features['resistance_strength'] = len([p for p in resistance_levels if abs(p - nearest_resistance) / nearest_resistance < 0.02])
            else:
                features['resistance_distance'] = 0.1
                features['resistance_strength'] = 0
            
            # サポートレベル
            support_levels = [price for _, price in low_peaks if price < current_price]
            if support_levels:
                nearest_support = max(support_levels, key=lambda x: -abs(x - current_price))
                features['support_distance'] = (current_price - nearest_support) / current_price
                features['support_strength'] = len([p for p in support_levels if abs(p - nearest_support) / nearest_support < 0.02])
            else:
                features['support_distance'] = 0.1
                features['support_strength'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"サポート・レジスタンス計算エラー: {e}")
            return {}
    
    def detect_regime_change(self, df: pd.DataFrame) -> Dict:
        """市場体制変化検出"""
        try:
            features = {}
            
            if len(df) < 100:
                return features
            
            returns = df['close'].pct_change().dropna()
            
            # ボラティリティ体制
            short_vol = returns.rolling(10).std().iloc[-1]
            long_vol = returns.rolling(50).std().iloc[-1]
            features['vol_regime'] = short_vol / long_vol if long_vol > 0 else 1.0
            
            # トレンド体制
            short_trend = df['close'].rolling(10).mean().iloc[-1]
            medium_trend = df['close'].rolling(30).mean().iloc[-1]
            long_trend = df['close'].rolling(50).mean().iloc[-1]
            
            features['trend_alignment'] = 0
            current_price = df['close'].iloc[-1]
            
            if current_price > short_trend > medium_trend > long_trend:
                features['trend_alignment'] = 1  # 強い上昇トレンド
            elif current_price < short_trend < medium_trend < long_trend:
                features['trend_alignment'] = -1  # 強い下降トレンド
            
            # 相関体制（市場全体との関係）
            features['correlation_regime'] = self.calculate_market_correlation_regime(returns)
            
            return features
            
        except Exception as e:
            logger.error(f"市場体制変化検出エラー: {e}")
            return {}
    
    def calculate_market_correlation_regime(self, returns: pd.Series) -> float:
        """市場相関体制計算"""
        try:
            # 簡易版：自己相関による体制判定
            if len(returns) < 30:
                return 0.0
            
            # ラグ1の自己相関
            autocorr_1 = returns.autocorr(lag=1)
            
            # ラグ5の自己相関
            autocorr_5 = returns.autocorr(lag=5)
            
            # 自己相関の変化率
            correlation_strength = (abs(autocorr_1) + abs(autocorr_5)) / 2
            
            return correlation_strength if not np.isnan(correlation_strength) else 0.0
            
        except Exception as e:
            logger.error(f"市場相関体制計算エラー: {e}")
            return 0.0
    
    def calculate_correlation_features(self, df: pd.DataFrame) -> Dict:
        """相関特徴量計算"""
        try:
            features = {}
            
            if len(df) < 50:
                return features
            
            # 価格と出来高の相関
            if 'volume' in df.columns:
                price_volume_corr = df['close'].corr(df['volume'])
                features['price_volume_correlation'] = price_volume_corr if not np.isnan(price_volume_corr) else 0.0
            
            # 高値・安値レンジの相関
            if len(df) >= 20:
                hl_range = (df['high'] - df['low']) / df['close']
                price_change = df['close'].pct_change()
                range_change_corr = hl_range.corr(price_change.abs())
                features['range_change_correlation'] = range_change_corr if not np.isnan(range_change_corr) else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"相関特徴量計算エラー: {e}")
            return {}

class MLModelManager:
    """機械学習モデル管理"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        if SKLEARN_AVAILABLE:
            self.initialize_models()
        
    def initialize_models(self):
        """モデル初期化"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        # XGBoostとLightGBMも利用可能な場合は追加
        try:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        except:
            pass
        
        try:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        except:
            pass
        
        # スケーラー初期化
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def prepare_training_data(self, feature_history: List[Dict], lookback_hours: int = 72) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """学習データ準備"""
        if len(feature_history) < lookback_hours + 24:  # 24時間後の結果が必要
            return None, None, None
        
        features_list = []
        labels_list = []
        
        for i in range(len(feature_history) - 24):
            if i + 24 >= len(feature_history):
                break
            
            current_features = feature_history[i]
            future_features = feature_history[i + 24]  # 24時間後
            
            if 'price' in current_features and 'price' in future_features:
                current_price = current_features['price']
                future_price = future_features['price']
                
                # ラベル作成（24時間後に上昇したかどうか）
                label = 1 if future_price > current_price else 0
                
                # 特徴量準備
                feature_vector = self.extract_feature_vector(current_features)
                if feature_vector is not None:
                    features_list.append(feature_vector)
                    labels_list.append(label)
        
        if len(features_list) < 10:  # 最小データ数
            return None, None, None
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        # 特徴量名取得
        feature_names = self.get_feature_names(feature_history[0] if feature_history else {})
        
        return features_array, labels_array, feature_names
    
    def extract_feature_vector(self, features: Dict) -> Optional[np.ndarray]:
        """特徴量ベクトル抽出"""
        try:
            # 重要な特徴量を選択
            key_features = [
                'price', 'returns_1h', 'returns_4h', 'returns_24h',
                'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
                'volatility_10', 'volatility_20', 'volatility_30',
                'rsi', 'rsi_slope', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'bb_width', 'volume_ratio',
                'fractal_dimension', 'hurst_exponent', 'entropy',
                'resistance_distance', 'support_distance',
                'vol_regime', 'trend_alignment'
            ]
            
            vector = []
            for feature in key_features:
                if feature in features:
                    value = features[feature]
                    if np.isnan(value) or np.isinf(value):
                        vector.append(0.0)
                    else:
                        vector.append(float(value))
                else:
                    vector.append(0.0)
            
            return np.array(vector)
            
        except Exception as e:
            logger.error(f"特徴量ベクトル抽出エラー: {e}")
            return None
    
    def get_feature_names(self, sample_features: Dict) -> List[str]:
        """特徴量名取得"""
        key_features = [
            'price', 'returns_1h', 'returns_4h', 'returns_24h',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
            'volatility_10', 'volatility_20', 'volatility_30',
            'rsi', 'rsi_slope', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width', 'volume_ratio',
            'fractal_dimension', 'hurst_exponent', 'entropy',
            'resistance_distance', 'support_distance',
            'vol_regime', 'trend_alignment'
        ]
        return key_features
    
    def train_models(self, features: np.ndarray, labels: np.ndarray, feature_names: List[str]):
        """モデル学習"""
        if not SKLEARN_AVAILABLE:
            logger.warning("機械学習ライブラリが利用できません")
            return
        
        try:
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"{model_name}モデルを学習中...")
                    
                    # データ正規化
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # モデル学習
                    model.fit(X_train_scaled, y_train)
                    
                    # 性能評価
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    
                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    self.model_performance[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    # 特徴量重要度
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = {
                            feature_names[i]: importance 
                            for i, importance in enumerate(model.feature_importances_)
                        }
                    
                    logger.info(f"{model_name}: 精度={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"{model_name}モデル学習エラー: {e}")
            
            # モデル保存
            self.save_models()
            
        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
    
    def predict_ensemble(self, features: Dict) -> Dict:
        """アンサンブル予測"""
        if not SKLEARN_AVAILABLE or not self.models:
            return {'probability': 0.5, 'confidence': 0.1, 'signal': 'HOLD'}
        
        try:
            feature_vector = self.extract_feature_vector(features)
            if feature_vector is None:
                return {'probability': 0.5, 'confidence': 0.1, 'signal': 'HOLD'}
            
            feature_vector = feature_vector.reshape(1, -1)
            
            predictions = []
            probabilities = []
            model_weights = []
            
            for model_name, model in self.models.items():
                try:
                    # データ正規化
                    feature_scaled = self.scalers[model_name].transform(feature_vector)
                    
                    # 予測
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(feature_scaled)[0][1]  # 上昇確率
                    else:
                        prob = 0.5
                    
                    pred = model.predict(feature_scaled)[0]
                    
                    predictions.append(pred)
                    probabilities.append(prob)
                    
                    # モデル重み（性能ベース）
                    if model_name in self.model_performance:
                        weight = self.model_performance[model_name]['cv_mean']
                    else:
                        weight = 0.5
                    
                    model_weights.append(weight)
                    
                except Exception as e:
                    logger.error(f"{model_name}予測エラー: {e}")
                    continue
            
            if not probabilities:
                return {'probability': 0.5, 'confidence': 0.1, 'signal': 'HOLD'}
            
            # 重み付きアンサンブル
            total_weight = sum(model_weights)
            if total_weight > 0:
                ensemble_prob = sum(p * w for p, w in zip(probabilities, model_weights)) / total_weight
            else:
                ensemble_prob = np.mean(probabilities)
            
            # 信頼度計算（予測の一致度）
            pred_agreement = sum(predictions) / len(predictions)
            confidence = 1.0 - 2 * abs(pred_agreement - 0.5)  # 0.5から離れるほど高信頼
            confidence = max(0.1, min(0.9, confidence))
            
            # シグナル判定
            if ensemble_prob > 0.65 and confidence > 0.4:
                signal = 'BUY'
            elif ensemble_prob < 0.35 and confidence > 0.4:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'probability': ensemble_prob,
                'confidence': confidence,
                'signal': signal,
                'model_predictions': dict(zip(self.models.keys(), probabilities)),
                'model_weights': dict(zip(self.models.keys(), model_weights))
            }
            
        except Exception as e:
            logger.error(f"アンサンブル予測エラー: {e}")
            return {'probability': 0.5, 'confidence': 0.1, 'signal': 'HOLD'}
    
    def save_models(self):
        """モデル保存"""
        try:
            for model_name, model in self.models.items():
                model_path = self.model_dir / f"{model_name}.pkl"
                scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[model_name], f)
            
            # 性能データ保存
            performance_path = self.model_dir / "performance.json"
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            # 特徴量重要度保存
            importance_path = self.model_dir / "feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            logger.info("モデル保存完了")
            
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
    
    def load_models(self):
        """モデル読み込み"""
        try:
            # 性能データ読み込み
            performance_path = self.model_dir / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            # 特徴量重要度読み込み
            importance_path = self.model_dir / "feature_importance.json"
            if importance_path.exists():
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            # モデル読み込み
            for model_name in list(self.models.keys()):
                model_path = self.model_dir / f"{model_name}.pkl"
                scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    
                    logger.info(f"{model_name}モデル読み込み完了")
                else:
                    logger.warning(f"{model_name}モデルファイルが見つかりません")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")

class AdvancedPredictionEngine:
    """高度予測エンジン"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_manager = MLModelManager()
        
        self.feature_history: List[Dict] = []
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5分
        
        # 学習設定
        self.min_training_samples = 200
        self.retrain_interval = 100  # 100回の予測後に再学習
        self.prediction_count = 0
        
        logger.info("高度予測エンジン初期化完了")
    
    def add_features(self, symbol: str, df: pd.DataFrame) -> Dict:
        """特徴量追加と履歴保存"""
        try:
            # テクニカル特徴量
            technical_features = self.feature_engineer.create_technical_features(df)
            
            # 高度特徴量
            advanced_features = self.feature_engineer.create_advanced_features(df)
            
            # 結合
            all_features = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                **technical_features,
                **advanced_features
            }
            
            # 履歴保存
            self.feature_history.append(all_features)
            
            # 履歴サイズ制限
            if len(self.feature_history) > 2000:
                self.feature_history = self.feature_history[-2000:]
            
            return all_features
            
        except Exception as e:
            logger.error(f"特徴量追加エラー: {e}")
            return {}
    
    def get_enhanced_prediction(self, symbol: str, df: pd.DataFrame) -> Dict:
        """強化予測"""
        try:
            # キャッシュチェック
            cache_key = f"{symbol}_{len(df)}"
            current_time = datetime.now().timestamp()
            
            if cache_key in self.prediction_cache:
                cached_time, cached_result = self.prediction_cache[cache_key]
                if current_time - cached_time < self.cache_timeout:
                    return cached_result
            
            # 特徴量作成
            features = self.add_features(symbol, df)
            if not features:
                return self._get_fallback_prediction(symbol)
            
            # 機械学習モデルによる予測
            ml_prediction = self.model_manager.predict_ensemble(features)
            
            # 従来の予測と組み合わせ
            traditional_prediction = self._get_traditional_prediction(features)
            
            # アンサンブル予測
            final_prediction = self._combine_predictions(ml_prediction, traditional_prediction)
            
            # キャッシュ保存
            self.prediction_cache[cache_key] = (current_time, final_prediction)
            
            # 予測回数カウント
            self.prediction_count += 1
            
            # 定期的な再学習
            if self.prediction_count % self.retrain_interval == 0:
                self._retrain_models()
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"強化予測エラー: {e}")
            return self._get_fallback_prediction(symbol)
    
    def _get_traditional_prediction(self, features: Dict) -> Dict:
        """従来の予測手法"""
        try:
            prediction_score = 0.0
            confidence_factors = []
            
            # 1. トレンド分析 (25%)
            if 'ma_ratio_20' in features:
                ma_ratio = features['ma_ratio_20']
                if ma_ratio > 1.02:
                    prediction_score += 0.15
                elif ma_ratio > 1.0:
                    prediction_score += 0.05
                elif ma_ratio < 0.98:
                    prediction_score -= 0.15
                else:
                    prediction_score -= 0.05
                confidence_factors.append(abs(ma_ratio - 1.0))
            
            # 2. モメンタム分析 (20%)
            if 'returns_1h' in features:
                momentum = features['returns_1h']
                if momentum > 0.01:
                    prediction_score += 0.10
                elif momentum > 0:
                    prediction_score += 0.05
                elif momentum < -0.01:
                    prediction_score -= 0.10
                else:
                    prediction_score -= 0.05
                confidence_factors.append(abs(momentum))
            
            # 3. RSI分析 (15%)
            if 'rsi' in features:
                rsi = features['rsi']
                if rsi < 30:
                    prediction_score += 0.10
                elif rsi > 70:
                    prediction_score -= 0.10
                confidence_factors.append(abs(rsi - 50) / 50)
            
            # 4. ボラティリティ調整 (10%)
            if 'volatility_10' in features:
                vol = features['volatility_10']
                if vol > 0.05:
                    prediction_score *= 0.9  # 高ボラティリティで控えめに
                confidence_factors.append(min(0.1, vol))
            
            # 5. 高度特徴量 (30%)
            if 'fractal_dimension' in features:
                fd = features['fractal_dimension']
                if fd > 1.5:
                    prediction_score += 0.05  # 複雑な動きは上昇を示唆
                elif fd < 1.3:
                    prediction_score -= 0.05
                confidence_factors.append(abs(fd - 1.5))
            
            if 'hurst_exponent' in features:
                hurst = features['hurst_exponent']
                if hurst > 0.6:
                    prediction_score += 0.08  # 持続性あり
                elif hurst < 0.4:
                    prediction_score -= 0.05  # ランダムウォーク
                confidence_factors.append(abs(hurst - 0.5))
            
            if 'entropy' in features:
                entropy = features['entropy']
                if entropy < 0.3:
                    prediction_score += 0.05  # 低エントロピー = 秩序
                elif entropy > 0.7:
                    prediction_score -= 0.05  # 高エントロピー = 混沌
                confidence_factors.append(1.0 - entropy)
            
            # 最終確率計算
            probability = 0.5 + prediction_score
            probability = max(0.1, min(0.9, probability))
            
            # 信頼度計算
            confidence = np.mean(confidence_factors) if confidence_factors else 0.3
            confidence = max(0.1, min(0.9, confidence))
            
            # シグナル判定
            if probability > 0.65 and confidence > 0.4:
                signal = 'BUY'
            elif probability < 0.35 and confidence > 0.4:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'probability': probability,
                'confidence': confidence,
                'signal': signal,
                'prediction_score': prediction_score
            }
            
        except Exception as e:
            logger.error(f"従来予測エラー: {e}")
            return {'probability': 0.5, 'confidence': 0.1, 'signal': 'HOLD'}
    
    def _combine_predictions(self, ml_pred: Dict, traditional_pred: Dict) -> Dict:
        """予測結合"""
        try:
            # 機械学習モデルの性能に基づく重み
            ml_weight = 0.7 if SKLEARN_AVAILABLE and self.model_manager.model_performance else 0.3
            traditional_weight = 1.0 - ml_weight
            
            # 確率の重み付き平均
            combined_prob = (
                ml_pred['probability'] * ml_weight +
                traditional_pred['probability'] * traditional_weight
            )
            
            # 信頼度の重み付き平均
            combined_confidence = (
                ml_pred['confidence'] * ml_weight +
                traditional_pred['confidence'] * traditional_weight
            )
            
            # シグナル決定（より厳格に）
            if combined_prob > 0.68 and combined_confidence > 0.5:
                signal = 'BUY'
            elif combined_prob < 0.32 and combined_confidence > 0.5:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'probability': combined_prob,
                'confidence': combined_confidence,
                'signal': signal,
                'ml_prediction': ml_pred,
                'traditional_prediction': traditional_pred,
                'weights': {'ml': ml_weight, 'traditional': traditional_weight}
            }
            
        except Exception as e:
            logger.error(f"予測結合エラー: {e}")
            return traditional_pred
    
    def _get_fallback_prediction(self, symbol: str) -> Dict:
        """フォールバック予測"""
        return {
            'probability': 0.5,
            'confidence': 0.1,
            'signal': 'HOLD',
            'symbol': symbol,
            'error': 'Insufficient data for prediction'
        }
    
    def _retrain_models(self):
        """モデル再学習"""
        try:
            if len(self.feature_history) < self.min_training_samples:
                logger.info(f"学習データ不足: {len(self.feature_history)}/{self.min_training_samples}")
                return
            
            logger.info("モデル再学習開始...")
            
            # 学習データ準備
            features, labels, feature_names = self.model_manager.prepare_training_data(self.feature_history)
            
            if features is not None and len(features) > 50:
                # モデル学習
                self.model_manager.train_models(features, labels, feature_names)
                logger.info("モデル再学習完了")
            else:
                logger.warning("学習データ準備失敗")
                
        except Exception as e:
            logger.error(f"モデル再学習エラー: {e}")
    
    def get_model_performance(self) -> Dict:
        """モデル性能取得"""
        return {
            'model_performance': self.model_manager.model_performance,
            'feature_importance': self.model_manager.feature_importance,
            'training_samples': len(self.feature_history),
            'prediction_count': self.prediction_count,
            'ml_available': SKLEARN_AVAILABLE
        }
    
    def force_retrain(self):
        """強制再学習"""
        self._retrain_models()

# テスト関数
def test_advanced_prediction():
    """高度予測エンジンテスト"""
    print("=== Advanced Prediction Engine Test ===")
    
    engine = AdvancedPredictionEngine()
    
    # サンプルデータ作成
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    volumes = np.random.lognormal(10, 0.5, 200)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 200)),
        'close': prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"テストデータ: {len(df)}行")
    
    # 予測実行
    for i in range(5):
        test_df = df.iloc[:100+i*20]
        prediction = engine.get_enhanced_prediction('BTC', test_df)
        
        print(f"予測 {i+1}: 確率={prediction['probability']:.3f}, "
              f"信頼度={prediction['confidence']:.3f}, "
              f"シグナル={prediction['signal']}")
    
    # パフォーマンス確認
    performance = engine.get_model_performance()
    print(f"\n性能統計:")
    print(f"学習サンプル数: {performance['training_samples']}")
    print(f"予測回数: {performance['prediction_count']}")
    print(f"ML利用可能: {performance['ml_available']}")
    
    if performance['model_performance']:
        print("\nモデル性能:")
        for model_name, perf in performance['model_performance'].items():
            print(f"  {model_name}: 精度={perf['accuracy']:.3f}")

if __name__ == "__main__":
    test_advanced_prediction()