#!/usr/bin/env python3
"""
分析生成的数据分布并与参考数据集比较
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

def load_and_analyze_data(file_path):
    """加载并分析单个数据文件"""
    print(f"分析文件: {file_path}")
    data = np.load(file_path)
    
    # 基本统计信息
    stats_info = {}
    for key in data.keys():
        arr = data[key]
        if key == 'samples':
            # 对于样本ID，只计算长度统计
            lengths = [len(s) for s in arr]
            stats_info[key] = {
                'shape': arr.shape,
                'dtype': arr.dtype,
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths)
            }
        else:
            # 对于数值数组，计算完整的统计信息
            stats_info[key] = {
                'shape': arr.shape,
                'dtype': arr.dtype,
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'nonzero_ratio': np.count_nonzero(arr) / arr.size
            }
    
    return stats_info

def plot_distributions(data, output_dir, tissue_name):
    """绘制数据分布图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{tissue_name} Data Distributions')
    
    # RNA表达分布
    if 'rna' in data:
        sns.histplot(data['rna'], bins=50, ax=axes[0,0])
        axes[0,0].set_title('RNA Expression Distribution')
        axes[0,0].set_xlabel('Expression Level')
        axes[0,0].set_ylabel('Count')
    
    # ATAC信号分布
    if 'atac' in data:
        atac_flat = data['atac'].flatten()
        sns.histplot(atac_flat, bins=50, ax=axes[0,1])
        axes[0,1].set_title('ATAC Signal Distribution')
        axes[0,1].set_xlabel('Signal Strength')
        axes[0,1].set_ylabel('Count')
    
    # 序列数据分布
    if 'sequence' in data:
        seq_flat = data['sequence'].flatten()
        sns.histplot(seq_flat, bins=4, ax=axes[1,0])
        axes[1,0].set_title('Sequence Distribution')
        axes[1,0].set_xlabel('Base Value')
        axes[1,0].set_ylabel('Count')
    
    # 样本分布
    if 'samples' in data:
        sample_lengths = [len(s) for s in data['samples']]
        sns.histplot(sample_lengths, bins=50, ax=axes[1,1])
        axes[1,1].set_title('Sample ID Length Distribution')
        axes[1,1].set_xlabel('Length')
        axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{tissue_name}_distributions.png')
    plt.close()

def plot_tissue_comparisons(tissue_data, output_dir):
    """绘制不同组织之间的数据分布比较"""
    output_dir = Path(output_dir)
    
    # 创建RNA表达比较图
    plt.figure(figsize=(12, 6))
    for tissue, data in tissue_data.items():
        sns.kdeplot(data['train_data']['rna'], label=f'{tissue} (train)')
    plt.title('RNA Expression Distribution Comparison')
    plt.xlabel('Expression Level')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'rna_distribution_comparison.png')
    plt.close()
    
    # 创建ATAC信号比较图
    plt.figure(figsize=(12, 6))
    for tissue, data in tissue_data.items():
        atac_flat = data['train_data']['atac'].flatten()
        sns.kdeplot(atac_flat[atac_flat > 0], label=f'{tissue} (train)')
    plt.title('ATAC Signal Distribution Comparison (Non-zero values)')
    plt.xlabel('Signal Strength')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'atac_distribution_comparison.png')
    plt.close()
    
    # 创建序列数据比较图
    plt.figure(figsize=(12, 6))
    base_labels = ['A', 'T', 'C', 'G', 'N']
    x = np.arange(len(base_labels))
    width = 0.2
    
    for i, (tissue, data) in enumerate(tissue_data.items()):
        seq_counts = np.bincount(data['train_data']['sequence'].flatten(), minlength=5)
        seq_freq = seq_counts / seq_counts.sum()
        plt.bar(x + i*width, seq_freq, width, label=tissue)
    
    plt.title('Sequence Base Distribution Comparison')
    plt.xlabel('Base')
    plt.ylabel('Frequency')
    plt.xticks(x + width, base_labels)
    plt.legend()
    plt.savefig(output_dir / 'sequence_distribution_comparison.png')
    plt.close()

def plot_atac_heatmap(data, output_dir, tissue_name):
    """绘制ATAC信号热图"""
    output_dir = Path(output_dir)
    
    # 选择前100个基因的ATAC信号进行可视化
    atac_subset = data['atac'][:100]
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(atac_subset, cmap='YlOrRd', xticklabels=False, yticklabels=False)
    plt.title(f'{tissue_name} ATAC Signal Heatmap (First 100 genes)')
    plt.xlabel('Position')
    plt.ylabel('Gene')
    plt.savefig(output_dir / f'{tissue_name}_atac_heatmap.png')
    plt.close()

def analyze_correlation(data, output_dir, tissue_name):
    """分析RNA表达和ATAC信号的相关性"""
    output_dir = Path(output_dir)
    
    # 计算每个基因的平均ATAC信号
    atac_mean = np.mean(data['atac'], axis=1)
    rna_expr = data['rna']
    
    # 计算相关系数
    correlation = np.corrcoef(atac_mean, rna_expr)[0,1]
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(atac_mean, rna_expr, alpha=0.5)
    plt.title(f'{tissue_name} RNA-ATAC Correlation (r={correlation:.3f})')
    plt.xlabel('Mean ATAC Signal')
    plt.ylabel('RNA Expression')
    plt.savefig(output_dir / f'{tissue_name}_correlation.png')
    plt.close()
    
    return correlation

def load_reference_data(ref_dir):
    """加载参考数据集"""
    print(f"加载参考数据集: {ref_dir}")
    ref_data = {
        'train': np.load(os.path.join(ref_dir, 'train_0.npz')),
        'test': np.load(os.path.join(ref_dir, 'test_0.npz'))
    }
    return ref_data

def compare_with_reference(generated_data, ref_data, output_dir):
    """比较生成的数据与参考数据"""
    output_dir = Path(output_dir)
    
    # 比较每个组织的数据
    for tissue in ['pbmc', 'brain', 'jejunum']:
        print(f"\n比较 {tissue} 数据...")
        
        # 比较训练集
        print("\n训练集比较:")
        for key in ['rna', 'atac', 'sequence']:
            gen_mean = np.mean(generated_data[tissue]['train_data'][key])
            ref_mean = np.mean(ref_data['train'][key])
            print(f"{key} 均值比较:")
            print(f"  生成数据: {gen_mean:.4f}")
            print(f"  参考数据: {ref_mean:.4f}")
            print(f"  差异: {abs(gen_mean - ref_mean):.4f}")
        
        # 绘制对比图
        plt.figure(figsize=(15, 5))
        
        # RNA表达对比
        plt.subplot(131)
        sns.kdeplot(generated_data[tissue]['train_data']['rna'], label='Generated')
        sns.kdeplot(ref_data['train']['rna'], label='Reference')
        plt.title(f'{tissue} RNA Expression')
        plt.legend()
        
        # ATAC信号对比
        plt.subplot(132)
        gen_atac = generated_data[tissue]['train_data']['atac'].flatten()
        ref_atac = ref_data['train']['atac'].flatten()
        sns.kdeplot(gen_atac[gen_atac > 0], label='Generated')
        sns.kdeplot(ref_atac[ref_atac > 0], label='Reference')
        plt.title(f'{tissue} ATAC Signal')
        plt.legend()
        
        # 序列分布对比
        plt.subplot(133)
        gen_seq = generated_data[tissue]['train_data']['sequence'].flatten()
        ref_seq = ref_data['train']['sequence'].flatten()
        sns.histplot(gen_seq, bins=4, alpha=0.5, label='Generated')
        sns.histplot(ref_seq, bins=4, alpha=0.5, label='Reference')
        plt.title(f'{tissue} Sequence')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{tissue}_comparison.png')
        plt.close()

def main():
    """主函数"""
    # 设置输出目录
    output_dir = Path('data/analysis_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载参考数据集
    ref_dir = '/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/data/26426404'
    ref_data = load_reference_data(ref_dir)
    
    # 要分析的组织
    tissues = ['pbmc', 'brain', 'jejunum']
    
    # 存储所有组织的数据
    tissue_data = {}
    
    # 分析每个组织的数据
    for tissue in tissues:
        print(f"\n分析 {tissue} 数据...")
        
        # 加载训练集和测试集
        train_data = np.load(f'data/improved_gene_view_datasets/{tissue}/train_0.npz')
        test_data = np.load(f'data/improved_gene_view_datasets/{tissue}/test_0.npz')
        
        # 存储数据以供后续比较
        tissue_data[tissue] = {
            'train_data': {k: v for k, v in train_data.items()},
            'test_data': {k: v for k, v in test_data.items()}
        }
        
        # 分析数据分布
        print("\n训练集统计信息:")
        train_stats = load_and_analyze_data(f'data/improved_gene_view_datasets/{tissue}/train_0.npz')
        for key, stats in train_stats.items():
            print(f"\n{key}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value}")
        
        print("\n测试集统计信息:")
        test_stats = load_and_analyze_data(f'data/improved_gene_view_datasets/{tissue}/test_0.npz')
        for key, stats in test_stats.items():
            print(f"\n{key}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value}")
        
        # 绘制分布图
        plot_distributions(train_data, output_dir, f'{tissue}_train')
        plot_distributions(test_data, output_dir, f'{tissue}_test')
        
        # 绘制ATAC热图
        plot_atac_heatmap(train_data, output_dir, f'{tissue}_train')
        
        # 分析RNA-ATAC相关性
        correlation = analyze_correlation(train_data, output_dir, f'{tissue}_train')
        print(f"\nRNA-ATAC相关系数: {correlation:.3f}")
        
        # 保存统计信息
        with open(output_dir / f'{tissue}_stats.txt', 'w') as f:
            f.write(f"{tissue} 数据统计信息\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("训练集统计信息:\n")
            for key, stats in train_stats.items():
                f.write(f"\n{key}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
            
            f.write("\n测试集统计信息:\n")
            for key, stats in test_stats.items():
                f.write(f"\n{key}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
            
            f.write(f"\nRNA-ATAC相关系数: {correlation:.3f}\n")
    
    # 绘制组织间比较图
    plot_tissue_comparisons(tissue_data, output_dir)
    
    # 与参考数据集比较
    compare_with_reference(tissue_data, ref_data, output_dir)
    
    print("\n分析完成！结果保存在", output_dir)

if __name__ == "__main__":
    main() 