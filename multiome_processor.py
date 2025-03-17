#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多组学数据处理模块 - 处理ATAC+RNA多组学数据
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

from utils import safe_plot_umap
from atac_processing import load_10x_atac, optimize_atac_preprocessing, annotate_peaks_to_genes, compute_gene_activity_matrix, compute_gene_activity_matrix_quick
from integration import integrate_rna_atac

def process_multiome_data_improved(tissue, h5_file, fragment_file=None, marker_genes=None, method='WNN', genome_annotation=None, peak_annotation_file=None, quick_test=False, resolution=0.8, memory_limit=0.8, auto_method_selection=False):
    """
    使用改进的ATAC整合方法处理多组学数据
    
    参数:
    - tissue: 组织类型名称
    - h5_file: 10x h5文件路径
    - fragment_file: ATAC fragments文件路径(可选)
    - marker_genes: 细胞类型标记基因字典
    - method: 整合方法，默认'WNN'
    - genome_annotation: 基因组注释文件路径(可选)
    - peak_annotation_file: 峰注释文件路径(可选)
    - quick_test: 是否使用快速测试模式，默认False
    - resolution: 聚类分辨率，默认0.8
    - memory_limit: 最大内存使用比例 (0.0-1.0)，默认0.8
    - auto_method_selection: 是否自动选择最佳整合方法
    
    返回:
    - 处理后的整合AnnData对象
    - 细胞类型计数
    """
    print(f"\n{'-'*80}")
    print(f"使用改进ATAC整合方法处理 {tissue} 多组学数据")
    print(f"{'-'*80}")
    
    # 监控内存使用
    import psutil
    process = psutil.Process(os.getpid())
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 读取RNA数据
    print(f"读取RNA数据: {h5_file}")
    try:
        # 尝试读取RNA数据
        adata = sc.read_10x_h5(h5_file)
        adata.var_names_make_unique()
        
        # 检查是否包含feature_types列，如果有则过滤RNA数据
        if 'feature_types' in adata.var:
            if 'Gene Expression' in adata.var['feature_types'].values:
                rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
            else:
                print(f"警告: 未找到'Gene Expression'类型，使用所有特征作为RNA数据")
                rna = adata.copy()
        else:
            print("警告: 未找到feature_types列，假设所有特征都是基因表达")
            rna = adata.copy()
        
        # 确保变量名唯一
        rna.var_names_make_unique()
        print(f"RNA数据: {rna.shape[0]} 细胞, {rna.shape[1]} 基因")
    except Exception as e:
        print(f"读取RNA数据时出错: {e}")
        return None, {}
    
    # 读取ATAC数据
    print(f"读取ATAC数据...")
    atac = load_10x_atac(h5_file, fragment_file)
    
    # RNA数据预处理
    print("预处理RNA数据...")
    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # 根据组织类型调整质控参数
    if tissue == 'pbmc':
        mt_threshold = 12
        min_genes = 200
    elif tissue == 'brain':
        mt_threshold = 10
        min_genes = 500
    else:  # jejunum或其他组织
        mt_threshold = 10
        min_genes = 300
    
    # 质量控制
    print(f"过滤线粒体比例 > {mt_threshold}% 的细胞...")
    rna = rna[rna.obs['pct_counts_mt'] < mt_threshold]
    print(f"过滤检测到基因数 < {min_genes} 的细胞...")
    sc.pp.filter_cells(rna, min_genes=min_genes)
    print(f"过滤在少于3个细胞中检测到的基因...")
    sc.pp.filter_genes(rna, min_cells=3)
    
    # 打印过滤后的细胞和基因数量
    print(f"过滤后RNA数据: {rna.shape[0]} 细胞, {rna.shape[1]} 基因")
    
    # 检查内存使用
    mem_usage = process.memory_info().rss / psutil.virtual_memory().total
    print(f"当前内存使用率: {mem_usage:.2%}")
    if mem_usage > memory_limit * 0.7:  # 如果已使用内存超过限制的70%
        print(f"警告: 内存使用接近限制 ({mem_usage:.2%}/{memory_limit:.2%})，尝试释放内存...")
        # 尝试释放一些内存
        import gc
        gc.collect()
    
    # 标准化和对数转换
    print("标准化RNA数据...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    # 鉴定高变异基因
    print("鉴定高变异基因...")
    sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=7, min_disp=0.5)
    print(f"找到 {sum(rna.var['highly_variable'])} 个高变异基因")
    
    # 检查内存使用
    mem_usage = process.memory_info().rss / psutil.virtual_memory().total
    print(f"当前内存使用率: {mem_usage:.2%}")
    
    # 计算PCA
    print("计算RNA数据的PCA...")
    n_pcs = min(50, rna.shape[1]-1)
    sc.pp.pca(rna, svd_solver='arpack', use_highly_variable=True, n_comps=n_pcs)
    
    # 计算RNA数据的邻域图和聚类(在没有ATAC数据时使用)
    print("计算RNA数据的邻域图...")
    sc.pp.neighbors(rna, n_neighbors=15, n_pcs=n_pcs)
    
    # 显示计算时间
    elapsed_time = time.time() - start_time
    print(f"RNA预处理完成，用时 {elapsed_time:.2f} 秒")
    
    # ATAC数据预处理和整合
    if atac is not None:
        print("预处理ATAC数据...")
        # 优化ATAC预处理
        atac = optimize_atac_preprocessing(atac, min_cells=5, min_peaks=200)
        
        # 注释peaks到基因
        print("注释peaks到基因...")
        atac = annotate_peaks_to_genes(atac, genomic_coords=genome_annotation, peak_annotation_file=peak_annotation_file)
        
        # 检查内存使用
        mem_usage = process.memory_info().rss / psutil.virtual_memory().total
        print(f"ATAC预处理后内存使用率: {mem_usage:.2%}")
        
        # 如果需要，计算基因活性矩阵
        gene_activity = None
        
        # 根据方法和自动选择标志确定是否需要计算基因活性矩阵
        need_gene_activity = (method in ['WNN', 'MNN'] or auto_method_selection)
        
        if need_gene_activity:
            print("计算ATAC基因活性矩阵...")
            # 获取RNA数据中的基因列表
            rna_genes = list(rna.var_names)
            
            # 根据是否快速测试选择合适的函数
            if quick_test:
                print("使用快速测试模式...")
                # 计算前检查内存
                if mem_usage > memory_limit * 0.8:
                    print("警告: 内存使用过高，将限制处理的基因数量...")
                    max_genes = 500  # 在内存紧张时进一步限制基因数量
                else:
                    max_genes = 1000
                
                gene_activity = compute_gene_activity_matrix_quick(atac, gene_list=rna_genes, max_genes=max_genes)
            else:
                # 检查内存，决定是否需要分块处理
                if mem_usage > memory_limit * 0.6:
                    print("警告: 内存使用较高，使用分块处理基因活性矩阵...")
                    # 实现分块处理逻辑，例如每次处理一部分基因
                    chunk_size = 1000
                    print(f"使用{chunk_size}基因的块大小...")
                    
                    # 分块处理
                    all_chunks = []
                    for i in range(0, len(rna_genes), chunk_size):
                        chunk_genes = rna_genes[i:i+chunk_size]
                        print(f"处理基因块 {i//chunk_size + 1}/{(len(rna_genes) + chunk_size - 1)//chunk_size}...")
                        chunk_activity = compute_gene_activity_matrix(atac, gene_list=chunk_genes)
                        if chunk_activity is not None:
                            all_chunks.append(chunk_activity)
                            
                        # 每处理完一个块，释放一些内存
                        gc.collect()
                    
                    # 合并所有块
                    if all_chunks:
                        # 可以在这里实现合并逻辑
                        # 简单情况下，可以使用第一个块的细胞，所有块的基因
                        # 这里需要根据具体情况调整
                        gene_activity = all_chunks[0]  # 暂时使用第一个块作为演示
                    else:
                        gene_activity = None
                else:
                    gene_activity = compute_gene_activity_matrix(atac, gene_list=rna_genes)
            
            if gene_activity is not None:
                print(f"基因活性矩阵: {gene_activity.shape[0]} 细胞, {gene_activity.shape[1]} 基因")
                # 在基因活性矩阵上计算LSI
                print("对基因活性矩阵进行TF-IDF变换...")
                from sklearn.feature_extraction.text import TfidfTransformer
                from sklearn.decomposition import TruncatedSVD
                
                gene_activity_tfidf = TfidfTransformer(norm='l2', use_idf=True).fit_transform(gene_activity.X)
                gene_activity.X = gene_activity_tfidf
                
                print("计算基因活性矩阵的SVD...")
                svd = TruncatedSVD(n_components=min(50, gene_activity.shape[1]-1), random_state=42)
                X_lsi = svd.fit_transform(gene_activity.X)
                gene_activity.obsm['X_lsi'] = X_lsi
                gene_activity.obsm['X_lsi_filtered'] = X_lsi[:, 1:]  # 去除第一个组件
        
        # 支持自动选择最佳方法
        if auto_method_selection:
            print("自动选择最佳整合方法...")
            from main import select_best_integration_method
            method = select_best_integration_method(rna, atac, gene_activity=gene_activity)
            print(f"选择的整合方法: {method}")
        
        # 整合RNA和ATAC数据
        print(f"使用 {method} 方法整合RNA和ATAC数据...")
        atac_for_integration = gene_activity if gene_activity is not None and method in ['WNN', 'MNN'] else atac
        integrated = integrate_rna_atac(rna, atac_for_integration, 
                                     use_gene_activity=(gene_activity is not None and method in ['WNN', 'MNN']),
                                     integration_method=method)
        
        if integrated is None:
            print("整合失败，仅使用RNA数据继续...")
            integrated = rna
    else:
        print("未找到ATAC数据，仅使用RNA数据...")
        integrated = rna
    
    # 检查内存使用
    mem_usage = process.memory_info().rss / psutil.virtual_memory().total
    print(f"整合后内存使用率: {mem_usage:.2%}")
    
    # 进行聚类
    print(f"使用分辨率 {resolution} 执行聚类...")
    try:
        sc.tl.leiden(integrated, resolution=resolution)
    except Exception as e:
        print(f"leiden聚类出错: {e}")
        print("尝试使用louvain聚类...")
        try:
            sc.tl.louvain(integrated, resolution=resolution)
            integrated.obs['leiden'] = integrated.obs['louvain']
        except Exception as e2:
            print(f"louvain聚类出错: {e2}")
            print("使用KMeans聚类...")
            from sklearn.cluster import KMeans
            X = integrated.obsm['X_pca'] if 'X_pca' in integrated.obsm else integrated.X
            
            # 如果有很多细胞，调整聚类数量
            if X.shape[0] > 10000:
                n_clusters = int(20 * (resolution / 0.8))  # 根据分辨率调整聚类数量
            else:
                n_clusters = int(10 * (resolution / 0.8))
                
            print(f"执行KMeans聚类，聚类数量: {n_clusters}...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            integrated.obs['leiden'] = [str(x) for x in kmeans.labels_]
    
    # 计算UMAP可视化
    try:
        print("计算UMAP可视化...")
        sc.tl.umap(integrated)
    except Exception as e:
        print(f"UMAP计算出错: {e}")
        print("跳过UMAP计算...")
    
    # 使用RNA数据进行差异表达分析
    try:
        print("执行差异表达分析...")
        if 'rna' in integrated.uns:
            rna_for_de = integrated.uns['rna']
        else:
            rna_for_de = rna
        
        rna_for_de.obs['leiden'] = integrated.obs['leiden']
        
        try:
            sc.tl.rank_genes_groups(rna_for_de, 'leiden', method='wilcoxon')
            print("差异表达分析完成")
        except Exception as e:
            print(f"差异表达分析出错: {e}")
    except Exception as e:
        print(f"准备差异表达分析时出错: {e}")
    
    # 使用标记基因注释细胞类型
    cell_type_counts = {}
    if marker_genes:
        print("根据标记基因注释细胞类型...")
        
        # 计算每个聚类的细胞类型评分
        cluster_marker_scores = {}
        for cluster in integrated.obs['leiden'].unique():
            try:
                if 'rna' in integrated.uns:
                    cluster_cells = integrated.uns['rna'][integrated.uns['rna'].obs['leiden'] == cluster]
                else:
                    # 修复索引不匹配问题
                    # 从集成数据中提取聚类细胞，然后基于细胞barcode在RNA中查找
                    cells_in_cluster = integrated.obs_names[integrated.obs['leiden'] == cluster]
                    # 确保RNA数据中存在这些细胞
                    common_cells = [cell for cell in cells_in_cluster if cell in rna.obs_names]
                    if len(common_cells) > 0:
                        cluster_cells = rna[common_cells]
                    else:
                        print(f"聚类{cluster}中没有找到在RNA数据中存在的细胞，跳过...")
                        cluster_marker_scores[cluster] = {}
                        continue
                scores = {}
                
                for cell_type, genes in marker_genes.items():
                    valid_genes = [g for g in genes if g in cluster_cells.var_names]
                    if valid_genes:
                        try:
                            gene_expressions = [np.mean(cluster_cells[:, g].X) for g in valid_genes]
                            scores[cell_type] = np.mean(gene_expressions)
                        except Exception as e:
                            print(f"计算{cell_type}评分时出错: {e}")
                            scores[cell_type] = 0
                    else:
                        scores[cell_type] = 0
                
                cluster_marker_scores[cluster] = scores
            except Exception as e:
                print(f"处理聚类{cluster}时出错: {e}")
                cluster_marker_scores[cluster] = {}
        
        # 分配细胞类型
        cluster_cell_types = {}
        for cluster, scores in cluster_marker_scores.items():
            if scores:
                max_score_cell_type = max(scores.items(), key=lambda x: x[1])[0] if scores else "Unknown"
                cluster_cell_types[cluster] = max_score_cell_type
            else:
                cluster_cell_types[cluster] = "Unknown"
        
        # 添加细胞类型注释
        integrated.obs['cell_type'] = [cluster_cell_types.get(clust, 'Unknown') for clust in integrated.obs['leiden']]
        
        # 计算细胞类型计数
        cell_type_counts = integrated.obs['cell_type'].value_counts().to_dict()
        
        print("\n细胞类型计数:")
        for cell_type, count in sorted(cell_type_counts.items()):
            print(f"  {cell_type}: {count}")
        
        # 保存结果
        output_file = f'data/processed/{tissue}_multiome_atac_improved_v2.h5ad'
        integrated.write(output_file)
        print(f"结果已保存到: {output_file}")
        
        # 导出细胞类型计数
        counts_df = pd.DataFrame(list(cell_type_counts.items()), columns=['cell_type', 'count'])
        counts_df['tissue'] = tissue
        counts_df.to_csv(f'data/processed/{tissue}_atac_improved_v2_cell_counts.csv', index=False)
        
        try:
            # 尝试绘制聚类可视化，使用安全绘图函数
            if 'X_umap' in integrated.obsm:
                safe_plot_umap(integrated, ['cell_type'], f"_{tissue}_cell_types.png")
        except Exception as e:
            print(f"绘图出错: {e}")
        
        return integrated, cell_type_counts
    
    print("无法执行细胞类型注释")
    return integrated, {}

    # 计算总运行时间
    elapsed_time = time.time() - start_time
    print(f"处理完成，总用时: {elapsed_time:.2f} 秒")
    
    return integrated, cell_type_counts 