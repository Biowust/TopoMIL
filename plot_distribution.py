import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_instance_importance_distribution(pos_scores, neg_scores, hard_scores, k=400, max_x_limits=(4000, 1000, 4000), save_dir='./'):

    os.makedirs(save_dir, exist_ok=True)
    
    pos_sorted = np.sort(pos_scores)[::-1]
    neg_sorted = np.sort(neg_scores)[::-1]
    hard_sorted = np.sort(hard_scores)[::-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    def plot_curve(ax, scores, title, color, x_limit):
        ax.plot(scores, color=color, linewidth=2.5, label='Instance Score')
        
        ax.axvline(x=k, color='red', linestyle='--', linewidth=2, label=f'Top-{k} Cutoff')
        ax.axvspan(0, k, color='red', alpha=0.1)
        

        ax.set_xlim(left=-50, right=x_limit)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Sorted Instance Index', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', fontsize=12)

    plot_curve(axes[0], pos_sorted, 'Positive Slide', '#1f77b4', max_x_limits[0])
    plot_curve(axes[1], neg_sorted, 'Negative Slide', '#2ca02c', max_x_limits[1])
    plot_curve(axes[2], hard_sorted, 'Hard Case Slide', '#ff7f0e', max_x_limits[2])

    axes[0].set_ylabel('Attention / Probability Score', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'custom_x_instance_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    dummy_pos = np.concatenate([np.random.uniform(0.8, 1.0, 15), np.random.uniform(0.0, 0.2, 1000)])
    
    dummy_neg = np.random.uniform(0.0, 0.15, 1000)
    
    dummy_hard = np.concatenate([np.random.uniform(0.4, 0.6, 50), np.random.uniform(0.1, 0.3, 900)])
    
    plot_instance_importance_distribution(
        pos_scores=dummy_pos, 
        neg_scores=dummy_neg, 
        hard_scores=dummy_hard, 
        k=10,
        save_dir='./'
    )