#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
モデル出力の詳細分析 - なぜ予測が小さいのか
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CryptoGAN
from src.utils import get_logger

logger = get_logger(__name__)

def analyze_model_outputs():
    """モデルの出力を詳細に分析"""
    
    # モデルの読み込み
    model_dir = Path('data/models_v2')
    
    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    model = CryptoGAN(
        input_size=metadata['input_size'],
        sequence_length=metadata['sequence_length']
    )
    
    # チェックポイントの読み込み
    checkpoint_path = list((model_dir / 'checkpoints').glob('*_generator.pth'))[-1]
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.generator.load_state_dict(checkpoint)
    
    model.generator.eval()
    
    # 様々な入力での出力を分析
    outputs_by_scale = []
    scales = np.linspace(0.1, 5.0, 20)
    
    for scale in scales:
        outputs = []
        for _ in range(100):
            noise = model.generate_noise(1) * scale
            hidden = model.generator.init_hidden(1)
            
            with torch.no_grad():
                output, _, _ = model.generator(noise, hidden)
                outputs.append(output.squeeze().numpy())
        
        outputs = np.array(outputs)
        outputs_by_scale.append({
            'scale': scale,
            'mean': np.mean(outputs),
            'std': np.std(outputs),
            'max': np.max(outputs),
            'min': np.min(outputs)
        })
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. スケール別の出力統計
    ax = axes[0, 0]
    scales_list = [d['scale'] for d in outputs_by_scale]
    means = [d['mean'] for d in outputs_by_scale]
    stds = [d['std'] for d in outputs_by_scale]
    
    ax.plot(scales_list, means, label='Mean Output')
    ax.fill_between(scales_list, 
                    [m-s for m,s in zip(means, stds)],
                    [m+s for m,s in zip(means, stds)],
                    alpha=0.3, label='±1 Std')
    ax.set_xlabel('Input Scale')
    ax.set_ylabel('Output Value')
    ax.set_title('Model Output vs Input Scale')
    ax.legend()
    ax.grid(True)
    
    # 2. 出力の分布（通常のスケール）
    ax = axes[0, 1]
    normal_outputs = []
    for _ in range(1000):
        noise = model.generate_noise(1)
        hidden = model.generator.init_hidden(1)
        with torch.no_grad():
            output, _, _ = model.generator(noise, hidden)
            normal_outputs.append(output.squeeze()[0].item())
    
    ax.hist(normal_outputs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Output Distribution (Mean: {np.mean(normal_outputs):.4f}, Std: {np.std(normal_outputs):.4f})')
    ax.grid(True, alpha=0.3)
    
    # 3. 実際のリターン分布との比較
    ax = axes[1, 0]
    
    # メタデータから実際のリターン統計を取得
    actual_mean = metadata['data_stats']['mean_return']
    actual_std = metadata['data_stats']['std_return']
    
    x = np.linspace(-0.1, 0.1, 100)
    actual_dist = (1/np.sqrt(2*np.pi*actual_std**2)) * np.exp(-0.5*((x-actual_mean)/actual_std)**2)
    model_std = np.std(normal_outputs)
    model_mean = np.mean(normal_outputs)
    model_dist = (1/np.sqrt(2*np.pi*model_std**2)) * np.exp(-0.5*((x-model_mean)/model_std)**2)
    
    ax.plot(x, actual_dist, label=f'Actual Returns (μ={actual_mean:.4f}, σ={actual_std:.4f})', linewidth=2)
    ax.plot(x, model_dist, label=f'Model Output (μ={model_mean:.4f}, σ={model_std:.4f})', linewidth=2)
    ax.set_xlabel('Return Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 推奨事項
    ax = axes[1, 1]
    ax.text(0.1, 0.9, "Analysis Results:", fontsize=14, weight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.8, f"Model output scale: {model_std:.6f}", transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Actual return scale: {actual_std:.6f}", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Scale ratio: {actual_std/model_std:.1f}x", transform=ax.transAxes)
    
    ax.text(0.1, 0.4, "Recommendations:", fontsize=12, weight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.3, "1. Scale model outputs by 10-100x", transform=ax.transAxes)
    ax.text(0.1, 0.2, "2. Use tanh activation for bounded outputs", transform=ax.transAxes)
    ax.text(0.1, 0.1, "3. Adjust loss function to penalize small outputs", transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/results/model_output_analysis.png')
    logger.info("Saved analysis to data/results/model_output_analysis.png")
    
    # 簡単な出力スケーリングのテスト
    logger.info("\n=== Output Scaling Test ===")
    scale_factors = [10, 50, 100]
    for scale_factor in scale_factors:
        scaled_outputs = [o * scale_factor for o in normal_outputs]
        scaled_std = np.std(scaled_outputs)
        logger.info(f"Scale factor {scale_factor}x: std = {scaled_std:.6f} (target: {actual_std:.6f})")

if __name__ == "__main__":
    analyze_model_outputs()