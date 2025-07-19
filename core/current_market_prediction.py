#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
現在の市場予測システム
改善されたモデルを使用してリアルタイム予測を実行
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 改善されたシステムをインポート
from improved_prediction_system import ImprovedPredictionSystem

class CurrentMarketPredictor:
    """現在の市場での予測システム"""
    
    def __init__(self):
        self.improved_system = ImprovedPredictionSystem()
        
    def get_latest_prediction(self):
        """最新の市場予測を取得"""
        logger.info("="*60)
        logger.info("現在の市場予測を開始")
        logger.info("="*60)
        
        # currentデータを使用して予測
        df = self.improved_system.load_period_data('current')
        if df is None or len(df) < 30:
            logger.error("データが不足しています")
            return None
            
        logger.info(f"データ期間: {df.index[0]} ～ {df.index[-1]}")
        logger.info(f"最新価格: ${df['close'].iloc[-1]:,.2f}")
        
        # 特徴量作成
        df = self.improved_system.create_advanced_features(df)
        if len(df) < 20:
            logger.error("特徴量作成後のデータが不足")
            return None
            
        # 特徴量とターゲット準備
        exclude_cols = ['target', 'target_direction', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values[:-1]  # 最後の行はターゲットがNaN
        
        # 欠損値処理
        X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
        
        # 特徴量選択
        X_selected, selected_indices = self.improved_system.select_best_features(
            X[:-1], y, k=min(50, X.shape[1])
        )
        
        # 予測用データ（最新の1行）
        X_latest = X[-1:, selected_indices]
        
        # 訓練データとテストデータ分割
        split_idx = int(0.8 * len(X_selected))
        X_train, X_val = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # スケーリング
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_latest_scaled = scaler.transform(X_latest)
        
        # アンサンブルモデル訓練
        base_models, meta_model, _ = self.improved_system.train_stacked_ensemble(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # 最新データで予測
        level1_latest = []
        for name, model in base_models.items():
            pred = model.predict(X_latest_scaled)
            level1_latest.append(pred[0])
        
        level1_latest = np.array(level1_latest).reshape(1, -1)
        final_prediction = meta_model.predict(level1_latest)[0]
        
        # 結果整理
        current_price = df['close'].iloc[-1]
        predicted_direction = "上昇" if final_prediction > 0 else "下降"
        confidence = abs(final_prediction)
        predicted_price_change = final_prediction * current_price
        predicted_next_price = current_price + predicted_price_change
        
        # 各モデルの予測
        model_predictions = {}
        for i, (name, model) in enumerate(base_models.items()):
            model_pred = model.predict(X_latest_scaled)[0]
            model_predictions[name] = {
                'prediction': float(model_pred),
                'direction': "上昇" if model_pred > 0 else "下降"
            }
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'predicted_return': float(final_prediction),
            'predicted_direction': predicted_direction,
            'confidence_score': float(confidence),
            'predicted_price_change': float(predicted_price_change),
            'predicted_next_price': float(predicted_next_price),
            'data_period': f"{df.index[0]} ～ {df.index[-1]}",
            'features_used': len(selected_indices),
            'model_consensus': model_predictions
        }
        
        return result
    
    def display_prediction(self, result):
        """予測結果を表示"""
        if result is None:
            print("X 予測を実行できませんでした")
            return
            
        print(f"\n{'='*60}")
        print("現在の市場予測結果")
        print(f"{'='*60}")
        print(f"予測時刻: {result['timestamp']}")
        print(f"現在価格: ${result['current_price']:,.2f}")
        print()
        
        print(f"予測方向: {result['predicted_direction']}")
        print(f"予測リターン: {result['predicted_return']:.4f} ({result['predicted_return']*100:.2f}%)")
        print(f"信頼度: {result['confidence_score']:.4f}")
        print(f"予測価格変動: ${result['predicted_price_change']:+,.2f}")
        print(f"予測価格: ${result['predicted_next_price']:,.2f}")
        print()
        
        print(f"使用特徴量数: {result['features_used']}")
        print(f"データ期間: {result['data_period']}")
        print()
        
        print("各モデルの予測:")
        print("-" * 40)
        for model_name, pred in result['model_consensus'].items():
            direction_symbol = "↑" if pred['direction'] == "上昇" else "↓"
            print(f"{direction_symbol} {model_name:15}: {pred['prediction']:+.4f} ({pred['direction']})")
        
        # 合意度計算
        up_votes = sum(1 for pred in result['model_consensus'].values() if pred['direction'] == "上昇")
        total_models = len(result['model_consensus'])
        consensus = up_votes / total_models
        
        print(f"\nモデル合意度:")
        if consensus > 0.7:
            print(f"   強い合意 ({up_votes}/{total_models} = {consensus:.1%}) - {result['predicted_direction']}")
        elif consensus > 0.6:
            print(f"   中程度の合意 ({up_votes}/{total_models} = {consensus:.1%}) - {result['predicted_direction']}")
        else:
            print(f"   弱い合意 ({up_votes}/{total_models} = {consensus:.1%}) - 慎重に判断を")
        
        print(f"\n{'='*60}")
        
        # リスク警告
        print("投資に関する重要な注意:")
        print("   - この予測は過去のデータに基づく統計的推定です")
        print("   - 仮想通貨は高いボラティリティを持ちます")
        print("   - 実際の投資判断には複数の情報源をご利用ください")
        print(f"{'='*60}")

def main():
    """メイン実行関数"""
    predictor = CurrentMarketPredictor()
    
    try:
        # 現在の市場予測を取得
        result = predictor.get_latest_prediction()
        
        # 結果表示
        predictor.display_prediction(result)
        
        # 結果をJSONで保存
        if result:
            with open('current_prediction.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"予測結果を current_prediction.json に保存しました")
            
    except Exception as e:
        logger.error(f"予測エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()