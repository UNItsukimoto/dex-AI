#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN訓練の診断と改善
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CryptoGAN
from src.utils import get_logger

logger = get_logger(__name__)

class GANDiagnostic:
    """GAN訓練の診断クラス"""
    
    def __init__(self, model_dir='data/models'):
        self.model_dir = Path(model_dir)
        
    def analyze_training_history(self):
        """訓練履歴の分析"""
        # 訓練履歴の読み込み（もし保存されていれば）
        history_path = self.model_dir / 'training_history.png'
        
        if history_path.exists():
            logger.info(f"Training history plot found: {history_path}")
        else:
            logger.warning("No training history found")
            
        # チェックポイントから損失を確認
        checkpoint_dir = self.model_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('gan_epoch_*_generator.pth'))
            
            losses = []
            for cp in checkpoints:
                epoch = int(cp.stem.split('_')[2])
                checkpoint = torch.load(cp, map_location='cpu')
                
                if 'metrics' in checkpoint:
                    losses.append({
                        'epoch': epoch,
                        'd_loss': checkpoint['metrics'].get('d_loss', None),
                        'g_loss': checkpoint['metrics'].get('g_loss', None)
                    })
            
            if losses:
                df_losses = pd.DataFrame(losses)
                self.plot_loss_analysis(df_losses)
                
    def plot_loss_analysis(self, df_losses):
        """損失の分析をプロット"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 損失の推移
        ax1.plot(df_losses['epoch'], df_losses['g_loss'], label='Generator Loss', marker='o')
        ax1.plot(df_losses['epoch'], df_losses['d_loss'], label='Discriminator Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('GAN Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損失の比率
        if df_losses['d_loss'].notna().any() and df_losses['g_loss'].notna().any():
            ratio = df_losses['g_loss'] / df_losses['d_loss']
            ax2.plot(df_losses['epoch'], ratio, marker='o', color='green')
            ax2.axhline(y=1, color='r', linestyle='--', label='Balanced (1:1)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('G/D Loss Ratio')
            ax2.set_title('Generator/Discriminator Balance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/gan_loss_analysis.png')
        logger.info("Saved GAN loss analysis")
        
    def check_mode_collapse(self, predictions):
        """モード崩壊の確認"""
        # 予測の多様性を確認
        unique_ratio = len(np.unique(predictions)) / len(predictions)
        std_dev = np.std(predictions)
        
        # 分布の確認
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. ヒストグラム
        ax = axes[0, 0]
        ax.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Prediction Distribution (Unique ratio: {unique_ratio:.3f})')
        ax.set_xlabel('Predicted Value')
        
        # 2. 時系列
        ax = axes[0, 1]
        ax.plot(predictions[:100])  # 最初の100個
        ax.set_title(f'First 100 Predictions (Std: {std_dev:.2f})')
        ax.set_xlabel('Time')
        
        # 3. 自己相関
        ax = axes[1, 0]
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(predictions[:min(len(predictions), 1000)], ax=ax, lags=40)
        ax.set_title('Autocorrelation')
        
        # 4. Q-Qプロット
        ax = axes[1, 1]
        stats.probplot(predictions, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig('data/results/mode_collapse_check.png')
        logger.info("Saved mode collapse analysis")
        
        # 診断結果
        if unique_ratio < 0.1:
            logger.warning("⚠️ Possible mode collapse detected! Very low prediction diversity.")
        elif unique_ratio < 0.3:
            logger.warning("⚠️ Limited prediction diversity. Model might be overfitting.")
        else:
            logger.info("✓ Prediction diversity seems reasonable.")
            
        return unique_ratio, std_dev

    def generate_synthetic_predictions(self, num_samples=1000):
        """合成予測の生成と分析"""
        # モデルの読み込み
        import json
        import pickle
        
        with open(f"{self.model_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
            
        gan = CryptoGAN(
            input_size=metadata['input_size'],
            sequence_length=metadata['sequence_length']
        )
        
        # チェックポイントの読み込み
        checkpoint_path = list((self.model_dir / 'checkpoints').glob('*_generator.pth'))[-1]
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            gan.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            gan.generator.load_state_dict(checkpoint)
            
        gan.generator.eval()
        
        # 合成データの生成
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples // metadata['sequence_length']):
                noise = gan.generate_noise(1)
                hidden = gan.generator.init_hidden(1)
                output, _, _ = gan.generator(noise, hidden)
                predictions.extend(output.squeeze().numpy())
        
        return np.array(predictions[:num_samples])

def diagnose_gan_issues():
    """GAN訓練の問題を診断"""
    diagnostic = GANDiagnostic()
    
    # 1. 訓練履歴の分析
    logger.info("Analyzing training history...")
    diagnostic.analyze_training_history()
    
    # 2. 合成予測の生成と分析
    logger.info("Generating synthetic predictions...")
    try:
        synthetic_predictions = diagnostic.generate_synthetic_predictions()
        
        # モード崩壊の確認
        logger.info("Checking for mode collapse...")
        unique_ratio, std_dev = diagnostic.check_mode_collapse(synthetic_predictions)
        
        # 推奨事項
        print("\n" + "="*60)
        print("GAN DIAGNOSTIC RESULTS")
        print("="*60)
        
        if unique_ratio < 0.3:
            print("\n⚠️ CRITICAL ISSUES FOUND:")
            print("1. Model shows signs of mode collapse")
            print("2. Predictions lack diversity")
            print("\nRECOMMENDED ACTIONS:")
            print("- Reduce learning rate (try 0.0001)")
            print("- Add noise to discriminator inputs")
            print("- Use label smoothing")
            print("- Implement gradient penalty")
            print("- Train discriminator more steps than generator")
        else:
            print("\n✓ Model diversity seems acceptable")
            
        print("\nNEXT STEPS:")
        print("1. Implement Wasserstein GAN (WGAN) for more stable training")
        print("2. Add feature matching loss")
        print("3. Use spectral normalization")
        print("4. Implement progressive training")
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_gan_issues()