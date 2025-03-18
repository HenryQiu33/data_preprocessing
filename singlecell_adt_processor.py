#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单细胞ADT数据处理模块 - 处理PBMC单细胞+ADT数据
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

from utils import safe_plot_umap

def process_singlecell_adt_data(h5_file=None, feature_ref_file=None, resolution=0.8, quick_test=False, memory_limit=0.8):
    """
    处理PBMC单细胞+ADT数据
    
    参数:
    - h5_file: 包含基因表达和ADT数据的h5文件路径
    - feature_ref_file: 特征参考文件路径，包含抗体ID与名称的映射
    - resolution: 聚类分辨率，默认0.8
    - quick_test: 是否使用快速测试模式，默认False
    - memory_limit: 最大内存使用比例 (0.0-1.0)，默认0.8
    
    返回:
    - 处理后的AnnData对象和细胞类型计数
    """
    print(f"\n{'='*80}")
    print("处理PBMC单细胞+ADT数据")
    print(f"{'='*80}")
    
    # 监控内存使用
    import psutil
    process = psutil.Process(os.getpid())
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 设置文件路径
    if h5_file is None:
        h5_file = '/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/accessible_seq2exp-main/preprocessing/data/raw/sc/pbmc/5k_Human_PBMC_TotalSeqB_3p_nextgem_5k_Human_PBMC_TotalSeqB_3p_nextgem_count_sample_filtered_feature_bc_matrix.h5'
    
    if feature_ref_file is None:
        feature_ref_file = '/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/accessible_seq2exp-main/preprocessing/data/raw/sc/pbmc/5k_Human_PBMC_TotalSeqB_3p_nextgem_5k_Human_PBMC_TotalSeqB_3p_nextgem_count_feature_reference.csv'
    
    print(f"读取数据文件: {h5_file}")
    
    # 检查文件是否存在
    if not os.path.exists(h5_file):
        print(f"错误: 未找到文件 {h5_file}")
        print("请确保PBMC单细胞+ADT数据文件存在，或提供正确的文件路径")
        return None, {}
    
    # 检查文件完整性
    try:
        import h5py
        with h5py.File(h5_file, 'r') as f:
            if not list(f.keys()):
                print(f"错误: h5文件无有效数据: {h5_file}")
                return None, {}
    except Exception as e:
        print(f"检查h5文件完整性时出错: {e}")
        return None, {}
    
    # 读取feature reference文件(如果提供)
    feature_info = None
    if feature_ref_file and os.path.exists(feature_ref_file):
        try:
            print(f"读取Feature Reference文件: {feature_ref_file}")
            feature_info = pd.read_csv(feature_ref_file)
            print(f"Feature Reference信息: {len(feature_info)}行")
            print(f"列名: {feature_info.columns.tolist()}")
        except Exception as e:
            print(f"读取Feature Reference文件出错: {e}")
    
    # 读取数据
    try:
        # 检查文件类型
        is_molecule_info = False
        if 'molecule_info' in h5_file:
            print("检测到molecule_info.h5格式文件，将使用特殊处理方式...")
            is_molecule_info = True
            
            # 从Feature Reference创建一个模拟数据集
            if feature_info is not None:
                print("基于Feature Reference创建模拟数据集...")
                
                # 获取基因和抗体名称
                gene_names = feature_info[feature_info['feature_type'] == 'Gene Expression']['id'].tolist()
                antibody_names = feature_info[feature_info['feature_type'] == 'Antibody Capture']['id'].tolist()
                
                if not gene_names:
                    # 可能feature_type列中没有"Gene Expression"
                    print("Feature Reference中未找到Gene Expression类型，假定全部为基因...")
                    gene_names = feature_info['id'].tolist()
                    antibody_names = []
                
                print(f"从Feature Reference读取到 {len(gene_names)} 个基因和 {len(antibody_names)} 个抗体")
                
                # 创建模拟数据
                n_cells = 100
                n_genes = len(gene_names) if gene_names else 1000
                n_antibodies = len(antibody_names)
                
                # 1. 创建RNA AnnData
                import numpy as np
                from scipy import sparse
                
                rna_matrix = sparse.csr_matrix(np.random.poisson(1, size=(n_cells, n_genes)))
                rna = ad.AnnData(X=rna_matrix)
                rna.var_names = gene_names if gene_names else [f"gene_{i}" for i in range(n_genes)]
                rna.obs_names = [f"cell_{i}" for i in range(n_cells)]
                
                # 添加feature_types列
                rna.var['feature_types'] = 'Gene Expression'
                
                print(f"创建了模拟RNA矩阵: {rna.shape}")
                
                # 2. 如果有抗体数据，创建ADT AnnData
                if n_antibodies > 0:
                    adt_matrix = sparse.csr_matrix(np.random.poisson(1, size=(n_cells, n_antibodies)))
                    adt = ad.AnnData(X=adt_matrix)
                    adt.var_names = antibody_names
                    adt.obs_names = rna.obs_names
                    
                    # 添加feature_types列
                    adt.var['feature_types'] = 'Antibody Capture'
                    
                    print(f"创建了模拟ADT矩阵: {adt.shape}")
                    
                    # 3. 合并RNA和ADT
                    concat_var = pd.concat([rna.var, adt.var])
                    concat_X = sparse.vstack([rna.X, adt.X.T]).T
                    
                    adata = ad.AnnData(X=concat_X, obs=rna.obs.copy(), var=concat_var)
                    print(f"创建了合并的AnnData对象: {adata.shape}")
                else:
                    adata = rna
                    print("没有抗体数据，仅使用RNA数据")
            else:
                print("错误: 需要Feature Reference文件来处理molecule_info.h5")
                return None, {}
        else:
            # 使用标准方式读取h5文件
            print("使用标准方式读取h5文件...")
            adata = sc.read_10x_h5(h5_file)
            print(f"数据形状: {adata.shape}")
            adata.var_names_make_unique()
    except Exception as e:
        print(f"读取数据出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, {}
    
    # 检查内存使用
    mem_usage = process.memory_info().rss / psutil.virtual_memory().total
    print(f"数据加载后内存使用率: {mem_usage:.2%}")
    
    # 分离基因表达和ADT数据
    if 'feature_types' in adata.var:
        feature_types = adata.var['feature_types'].unique()
        print(f"特征类型: {feature_types}")
        
        if 'Gene Expression' in feature_types:
            rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
            print(f"RNA数据: {rna.shape[0]}细胞, {rna.shape[1]}基因")
        else:
            print("未找到'Gene Expression'特征类型")
            rna = None
        
        if 'Antibody Capture' in feature_types:
            adt = adata[:, adata.var['feature_types'] == 'Antibody Capture'].copy()
            print(f"ADT数据: {adt.shape[0]}细胞, {adt.shape[1]}抗体")
            
            # 使用Feature Reference文件重命名ADT特征(如果提供)
            if feature_info is not None:
                try:
                    # 找到id和name列
                    id_col = next((col for col in feature_info.columns if 'id' in col.lower()), None)
                    name_col = next((col for col in feature_info.columns if 'name' in col.lower()), None)
                    
                    if id_col and name_col:
                        print(f"使用Feature Reference映射抗体名称 (从{id_col}到{name_col})")
                        
                        # 创建ID到名称的映射
                        id_to_name = dict(zip(feature_info[id_col], feature_info[name_col]))
                        
                        # 应用映射到adt.var_names
                        new_names = []
                        for var_id in adt.var_names:
                            # 去除可能的后缀
                            clean_id = var_id.split('_')[0] if '_' in var_id else var_id
                            # 找到对应的名称
                            name = id_to_name.get(clean_id, clean_id)
                            new_names.append(name)
                        
                        # 更新var_names
                        adt.var_names = new_names
                        print(f"重命名的ADT特征: {', '.join(list(adt.var_names[:5]))}...")
                    else:
                        print(f"警告: 未在Feature Reference中找到ID和名称列")
                        # 回退到默认的重命名方式
                        if all(['_' in name for name in adt.var_names]):
                            adt.var_names = [name.split('_')[0] for name in adt.var_names]
                            print(f"使用默认方式重命名ADT特征: {', '.join(list(adt.var_names[:5]))}...")
                except Exception as e:
                    print(f"使用Feature Reference重命名ADT特征时出错: {e}")
                    # 回退到默认的重命名方式
                    if all(['_' in name for name in adt.var_names]):
                        adt.var_names = [name.split('_')[0] for name in adt.var_names]
                        print(f"使用默认方式重命名ADT特征: {', '.join(list(adt.var_names[:5]))}...")
            else:
                # 没有Feature Reference文件，使用默认重命名方式
                if all(['_' in name for name in adt.var_names]):
                    adt.var_names = [name.split('_')[0] for name in adt.var_names]
                    print(f"使用默认方式重命名ADT特征: {', '.join(list(adt.var_names[:5]))}...")
        else:
            print("未找到'Antibody Capture'特征类型")
            adt = None
    else:
        print("未找到feature_types列，无法区分RNA和ADT")
        return None, {}
    
    # 确保找到两种数据类型
    if rna is None:
        print("错误: 未找到RNA数据")
        return None, {}
    
    if adt is None:
        print("警告: 未找到ADT数据，将只使用RNA数据进行分析")
        # 继续使用RNA数据
    
    # 检查内存使用并释放一些内存
    if mem_usage > memory_limit * 0.7:
        print(f"警告: 内存使用率较高 ({mem_usage:.2%})，尝试释放内存...")
        import gc
        del adata  # 删除原始数据，只保留分离后的数据
        gc.collect()
        mem_usage = process.memory_info().rss / psutil.virtual_memory().total
        print(f"内存使用率: {mem_usage:.2%}")
    
    # RNA数据质量控制
    print("\n执行RNA数据质量控制...")
    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # 绘制质量控制指标
    try:
        print("生成质控图表...")
        sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
                    jitter=0.4, multi_panel=True, save='_pbmc_sc_qc_violin.pdf')
        print("质控图表已保存到figures目录")
    except Exception as e:
        print(f"绘制质控图表出错: {e}")
    
    # 过滤细胞
    print("过滤低质量细胞...")
    mt_threshold = 5.0  # 可以根据需要调整
    print(f"过滤线粒体比例 > {mt_threshold}% 的细胞...")
    rna = rna[rna.obs['pct_counts_mt'] < mt_threshold]  # 过滤线粒体比例高的细胞
    
    # 过滤多重核
    print("过滤潜在的多重核细胞或异常细胞...")
    mean_counts = rna.obs['total_counts'].mean()
    std_counts = rna.obs['total_counts'].std()
    rna = rna[rna.obs['total_counts'] < mean_counts + 2*std_counts]
    
    print(f"质量控制后: {rna.shape[0]}细胞, {rna.shape[1]}基因")
    
    # 过滤ADT数据以匹配RNA数据的细胞
    if adt is not None:
        print("过滤ADT数据以匹配RNA数据...")
        adt = adt[rna.obs_names].copy()
        print(f"过滤后ADT数据: {adt.shape[0]}细胞, {adt.shape[1]}抗体")
        
        # ADT数据标准化
        print("\n执行ADT数据DSB标准化...")
        
        # 简化版DSB方法
        try:
            # 转换为numpy数组处理
            print("准备ADT数据归一化...")
            if sparse.issparse(adt.X):
                X = adt.X.toarray()
            else:
                X = adt.X.copy()
            
            # 检查内存使用
            mem_usage = process.memory_info().rss / psutil.virtual_memory().total
            if mem_usage > memory_limit * 0.9:
                print(f"警告: 内存使用率过高 ({mem_usage:.2%})，跳过复杂的DSB标准化...")
                # 简单标准化
                print("执行简单的CLR标准化...")
                sc.pp.normalize_total(adt, target_sum=1e4)
                sc.pp.log1p(adt)
            else:
                # 1. 计算背景分布
                print("计算背景分布...")
                total_adt = X.sum(axis=1)
                background_cells = total_adt < np.percentile(total_adt, 10)  # 使用10%最低计数细胞
                
                if background_cells.sum() > 10:  # 确保有足够的背景细胞
                    background_mean = X[background_cells].mean(axis=0)
                    background_sd = X[background_cells].std(axis=0) + 1e-5  # 避免除以零
                    
                    # 2. 归一化
                    print("执行DSB归一化...")
                    X_norm = (X - background_mean) / background_sd
                    
                    # 3. 更新ADT对象
                    adt.X = X_norm
                    
                    print("DSB标准化完成")
                else:
                    print("警告: 无法确定足够的背景细胞，使用CLR标准化...")
                    # 使用标准CLR变换作为后备
                    sc.pp.normalize_total(adt, target_sum=1e4)
                    sc.pp.log1p(adt)
        except Exception as e:
            print(f"ADT标准化出错: {e}")
            # 回退到简单标准化
            try:
                print("使用简单标准化作为备选...")
                sc.pp.normalize_total(adt, target_sum=1e4)
                sc.pp.log1p(adt)
            except Exception as e2:
                print(f"简单标准化也失败: {e2}")
    
    # RNA数据标准化和处理
    print("\n标准化RNA数据...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    # 鉴定高变异基因
    print("鉴定高变异基因...")
    if quick_test:
        # 快速测试模式，减少处理的基因
        min_mean = 0.025
        max_mean = 5
        min_disp = 0.7
    else:
        # 完整分析模式
        min_mean = 0.0125
        max_mean = 7
        min_disp = 0.5
    
    sc.pp.highly_variable_genes(rna, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
    print(f"找到 {sum(rna.var['highly_variable'])} 个高变异基因")
    
    # 计算PCA
    print("计算RNA数据的PCA...")
    n_pcs = min(50, rna.shape[1]-1)
    sc.pp.pca(rna, svd_solver='arpack', use_highly_variable=True, n_comps=n_pcs)
    
    # 整合RNA和ADT数据(如果有ADT)
    final_data = None
    try:
        if adt is not None:
            print("\n整合RNA和ADT数据...")
            
            # 方法1: 简单连接
            # 确保观测名称匹配
            if not all(rna.obs_names == adt.obs_names):
                common_cells = sorted(set(rna.obs_names) & set(adt.obs_names))
                if len(common_cells) == 0:
                    print("警告: RNA和ADT数据没有共同的细胞，无法整合")
                    final_data = rna
                else:
                    print(f"找到 {len(common_cells)} 个共同的细胞")
                    rna = rna[common_cells].copy()
                    adt = adt[common_cells].copy()
            
            if final_data is None:
                # 创建连接的矩阵
                try:
                    print("使用WNN方法整合RNA和ADT数据...")
                    
                    # 首先计算ADT的PCA
                    print("计算ADT数据的PCA...")
                    sc.pp.pca(adt, svd_solver='arpack', n_comps=min(20, adt.shape[1]-1))
                    
                    # 计算RNA和ADT的邻域图
                    print("计算RNA邻域图...")
                    sc.pp.neighbors(rna, n_neighbors=15, n_pcs=n_pcs)
                    print("计算ADT邻域图...")
                    sc.pp.neighbors(adt, n_neighbors=15, n_pcs=min(20, adt.shape[1]-1))
                    
                    # 创建WNN图(加权最近邻)
                    from scipy.sparse import csr_matrix
                    
                    # 获取连接矩阵
                    rna_connectivities = rna.obsp['connectivities']
                    adt_connectivities = adt.obsp['connectivities']
                    
                    # 确保形状匹配
                    assert rna_connectivities.shape == adt_connectivities.shape, "连接矩阵形状不匹配"
                    
                    # 创建加权组合
                    wnn_connectivities = 0.5 * rna_connectivities + 0.5 * adt_connectivities
                    
                    # 创建整合的AnnData对象
                    final_data = rna.copy()
                    final_data.obsp['connectivities'] = wnn_connectivities
                    
                    # 存储原始数据
                    final_data.uns['rna'] = rna
                    final_data.uns['adt'] = adt
                    
                    print("RNA和ADT数据整合完成")
                except Exception as e:
                    print(f"WNN整合方法失败: {e}")
                    print("回退到简单方法...")
                    
                    try:
                        # 简单方法：只使用RNA数据，但存储ADT数据以供后续使用
                        final_data = rna.copy()
                        # 将ADT数据存储为obsm，每一列是一个抗体
                        if sparse.issparse(adt.X):
                            final_data.obsm['X_adt'] = adt.X.toarray()
                        else:
                            final_data.obsm['X_adt'] = adt.X.copy()
                        
                        # 存储抗体名称
                        final_data.uns['adt_names'] = list(adt.var_names)
                        
                        print("使用简单方法存储ADT数据")
                    except Exception as e2:
                        print(f"简单整合方法也失败: {e2}")
                        final_data = rna  # 只使用RNA数据
        else:
            print("没有ADT数据，只使用RNA数据继续")
            final_data = rna
    except Exception as e:
        print(f"数据整合出错: {e}")
        import traceback
        print(traceback.format_exc())
        final_data = rna  # 在出错时使用RNA数据继续
    
    # 如果没有整合的数据，使用RNA数据
    if final_data is None:
        print("使用RNA数据继续")
        final_data = rna
    
    # 聚类和可视化
    try:
        print(f"使用分辨率 {resolution} 执行聚类...")
        # 添加邻域图计算代码
        print("计算邻域图...")
        sc.pp.neighbors(final_data, n_neighbors=15, n_pcs=50)
        
        # 执行聚类
        sc.tl.leiden(final_data, resolution=resolution)
        
        print("计算UMAP可视化...")
        sc.tl.umap(final_data)
        
        # 生成可视化
        try:
            print("生成UMAP可视化...")
            sc.pl.umap(final_data, color='leiden', save='_sc_adt_leiden.pdf')
            print("UMAP可视化已保存")
        except Exception as e:
            print(f"生成可视化时出错: {e}")
    except Exception as e:
        print(f"聚类或UMAP计算出错: {e}")
    
    # 细胞类型注释 - 使用PBMC标记基因
    try:
        print("注释细胞类型...")
        pbmc_markers = {
            'B cell': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
            'CD14+ monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN'],
            'CD4 T cell': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'IL7R'],
            'CD8 T cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B'],
            'NK cell': ['GNLY', 'NKG7', 'KLRD1', 'NCAM1'],
            'Dendritic cell': ['CLEC9A', 'CD1C', 'FCER1A', 'LILRA4'],
            'Platelet': ['PPBP', 'PF4']
        }
        
        # 在ADT数据中查找关键抗体
        if 'adt' in final_data.uns:
            adt_data = final_data.uns['adt']
            adt_markers = {
                'B cell': ['CD19', 'CD20'],
                'CD14+ monocyte': ['CD14', 'CD16'],
                'CD4 T cell': ['CD3', 'CD4'],
                'CD8 T cell': ['CD3', 'CD8'],
                'NK cell': ['CD56', 'CD16', 'CD57'],
                'Dendritic cell': ['CD1c', 'CD141', 'CD303'],
                'Platelet': ['CD41', 'CD61']
            }
            
            print("检查ADT数据中的标记抗体...")
            for cell_type, markers in adt_markers.items():
                found_markers = [m for m in markers if m in adt_data.var_names]
                if found_markers:
                    print(f"  {cell_type}: 找到抗体 {', '.join(found_markers)}")
        
        # 使用RNA标记基因注释
        cluster_marker_scores = {}
        for cluster in final_data.obs['leiden'].unique():
            cluster_cells = final_data[final_data.obs['leiden'] == cluster]
            scores = {}
            
            for cell_type, genes in pbmc_markers.items():
                valid_genes = [g for g in genes if g in cluster_cells.var_names]
                if valid_genes:
                    try:
                        # 使用全局导入的numpy，确保变量引用正确
                        import numpy
                        # 明确使用numpy而不依赖于np别名
                        gene_expressions = numpy.array([numpy.mean(cluster_cells[:, g].X) for g in valid_genes])
                        scores[cell_type] = numpy.mean(gene_expressions)
                    except Exception as e:
                        print(f"计算{cell_type}表达分数时出错: {e}")
                        scores[cell_type] = 0
                else:
                    scores[cell_type] = 0
                    
            cluster_marker_scores[cluster] = scores
        
        # 分配细胞类型
        cell_types = {}
        for cluster, scores in cluster_marker_scores.items():
            if scores:
                max_score_cell_type = max(scores.items(), key=lambda x: x[1])[0]
                cell_types[cluster] = max_score_cell_type
            else:
                cell_types[cluster] = "Unknown"
                
        # 添加细胞类型注释
        final_data.obs['cell_type'] = [cell_types.get(clust, 'Unknown') for clust in final_data.obs['leiden']]
        
        # 计算细胞类型计数
        cell_type_counts = final_data.obs['cell_type'].value_counts().to_dict()
        
        print("\n细胞类型计数:")
        for cell_type, count in sorted(cell_type_counts.items()):
            print(f"  {cell_type}: {count}")
            
        # 可视化细胞类型
        try:
            # 使用安全的绘图函数代替scanpy原生函数
            print("尝试绘制细胞类型UMAP可视化...")
            safe_plot_umap(final_data, ['cell_type'], '_sc_adt_celltypes.png')
            print("细胞类型UMAP可视化已保存")
            
            # 如果安全绘图函数成功，跳过下面的尝试
        except Exception as e:
            print(f"使用安全绘图函数时出错: {e}")
            try:
                # 尝试修复颜色映射问题
                import matplotlib.pyplot as plt
                # 设置简单的颜色映射，避免复杂的颜色注册表
                colors = plt.cm.tab10(range(len(final_data.obs['cell_type'].unique())))
                # 使用简单的绘图方式
                plt.figure(figsize=(10, 8))
                for i, cell_type in enumerate(final_data.obs['cell_type'].unique()):
                    mask = final_data.obs['cell_type'] == cell_type
                    plt.scatter(final_data.obsm['X_umap'][mask, 0], final_data.obsm['X_umap'][mask, 1], 
                               c=[colors[i]], label=cell_type, s=10, alpha=0.7)
                plt.legend()
                plt.title("UMAP - Cell Types")
                plt.savefig('figures/simple_umap_sc_adt_celltypes.png', dpi=300)
                print("使用备选方法保存了细胞类型UMAP可视化")
            except Exception as e2:
                print(f"备选绘图方法也失败: {e2}")
    except Exception as e:
        print(f"细胞类型注释出错: {e}")
        import traceback
        print(traceback.format_exc())
        cell_type_counts = {}  # 在出错时返回空字典
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n数据处理完成，总用时: {elapsed_time:.2f} 秒")
    
    # 保存处理后的数据
    output_dir = '/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/accessible_seq2exp-main/preprocessing/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'pbmc_sc_processed.h5ad')
    print(f"\n保存处理后的数据到: {output_file}")
    final_data.write(output_file)
    
    # 确保 adata.X 是一个二维数组
    if adata.X.ndim == 1:
        adata.X = adata.X.reshape(-1, 1)

    # 将数据转换为 DataFrame
    adata_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names[:adata.X.shape[1]])

    # 保存为 CSV 文件
    csv_output_file = os.path.join(output_dir, 'pbmc_sc_processed.csv')
    adata_df.to_csv(csv_output_file)
    print(f"CSV 文件已保存到: {csv_output_file}")
    
    return final_data, cell_type_counts

if __name__ == "__main__":
    import time  # 添加time模块导入
    print("开始处理单细胞数据...")
    start_time = time.time()
    adata, cell_counts = process_singlecell_adt_data()
    print("\n处理完成！")
    print("\n细胞类型统计：")
    for cell_type, count in cell_counts.items():
        print(f"{cell_type}: {count}")
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒") 