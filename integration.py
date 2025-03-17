#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据整合模块 - RNA和ATAC数据的整合方法
"""

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
import gc
import time
import traceback
warnings.filterwarnings('ignore')

def integrate_rna_atac(rna, atac, n_pcs=30, use_gene_activity=True, integration_method='WNN', memory_limit=0.8):
    """
    整合RNA和ATAC数据
    
    参数:
    - rna: AnnData对象，包含RNA数据
    - atac: AnnData对象，包含ATAC数据
    - n_pcs: 用于整合的主成分数量
    - use_gene_activity: 是否使用基因活性矩阵
    - integration_method: 整合方法，'WNN'或'Concatenate'或'MNN'或'Harmony'
    - memory_limit: 最大内存使用比例 (0.0-1.0)，默认0.8
    
    返回:
    - 整合后的AnnData对象
    """
    start_time = time.time()
    print(f"整合RNA({rna.shape})和ATAC({atac.shape})数据...")
    
    # 监控内存使用
    try:
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_info().rss / psutil.virtual_memory().total
        print(f"整合开始时内存使用率: {mem_usage:.2%}")
    except:
        print("无法监控内存使用")
    
    # 确保细胞ID匹配
    common_cells = list(set(rna.obs_names).intersection(atac.obs_names))
    if len(common_cells) == 0:
        print("错误: RNA和ATAC数据没有共同的细胞")
        return None
    
    print(f"找到{len(common_cells)}个共同细胞")
    rna_subset = rna[common_cells].copy()
    atac_subset = atac[common_cells].copy()
    
    # 确保两个数据集都有降维结果
    if 'X_pca' not in rna_subset.obsm:
        print("计算RNA PCA...")
        try:
            sc.pp.pca(rna_subset, n_comps=min(n_pcs, rna_subset.shape[1]-1))
        except Exception as e:
            print(f"RNA PCA计算失败: {e}")
            print("尝试使用随机降维...")
            rna_subset.obsm['X_pca'] = TruncatedSVD(
                n_components=min(n_pcs, rna_subset.shape[1]-1), 
                random_state=42
            ).fit_transform(rna_subset.X)
    
    atac_rep_key = 'X_lsi' if use_gene_activity else 'X_lsi'
    if atac_rep_key not in atac_subset.obsm:
        print(f"计算ATAC {atac_rep_key}...")
        if not use_gene_activity:
            # 在原始ATAC数据上计算LSI
            try:
                svd = TruncatedSVD(n_components=min(n_pcs, atac_subset.shape[1]-1), random_state=42)
                X_lsi = svd.fit_transform(atac_subset.X)
                atac_subset.obsm['X_lsi'] = X_lsi
                atac_subset.obsm['X_lsi_filtered'] = X_lsi[:, 1:]  # 移除第一个成分
            except Exception as e:
                print(f"ATAC LSI计算失败: {e}")
                print("尝试使用随机降维...")
                atac_subset.obsm['X_lsi'] = TruncatedSVD(
                    n_components=min(n_pcs, atac_subset.shape[1]-1), 
                    random_state=42
                ).fit_transform(atac_subset.X)
                atac_subset.obsm['X_lsi_filtered'] = atac_subset.obsm['X_lsi'][:, 1:]
    
    try:
        # 检查内存使用并尝试优化
        try:
            mem_usage = process.memory_info().rss / psutil.virtual_memory().total
            print(f"降维后内存使用率: {mem_usage:.2%}")
            if mem_usage > memory_limit * 0.7:
                print(f"警告: 内存使用接近限制 ({mem_usage:.2%}/{memory_limit:.2%})，尝试释放内存...")
                # 尝试释放一些内存
                gc.collect()
                # 如果内存仍然紧张，转向更高效的整合方法
                if mem_usage > memory_limit * 0.9:
                    print("内存紧张，切换到更高效的Concatenate方法...")
                    integration_method = 'Concatenate'
        except:
            pass
        
        # 处理方法: Weighted Nearest Neighbors (WNN)
        if integration_method == 'WNN':
            print("使用WNN方法整合数据")
            
            try:
                # 创建两个模态的KNN图
                print("计算RNA邻域图...")
                rna_neighbors = NearestNeighbors(n_neighbors=20).fit(rna_subset.obsm['X_pca'])
                rna_dist, rna_indices = rna_neighbors.kneighbors()
                
                print("计算ATAC邻域图...")
                atac_rep = atac_subset.obsm['X_lsi_filtered'] if 'X_lsi_filtered' in atac_subset.obsm else atac_subset.obsm['X_lsi']
                atac_neighbors = NearestNeighbors(n_neighbors=20).fit(atac_rep)
                atac_dist, atac_indices = atac_neighbors.kneighbors()
                
                # 归一化距离矩阵
                print("归一化距离矩阵...")
                rna_dist = rna_dist / np.max(rna_dist)
                atac_dist = atac_dist / np.max(atac_dist)
                
                # 计算加权图
                print("创建加权图...")
                alpha = 0.5  # RNA和ATAC的权重
                combined_dist = alpha * rna_dist + (1 - alpha) * atac_dist
                
                # 基于图的距离重新构建KNN索引
                from scipy.sparse import csr_matrix
                
                # 创建稀疏连接矩阵
                n_cells = len(common_cells)
                rows = np.repeat(np.arange(n_cells), rna_indices.shape[1])
                cols = rna_indices.flatten()
                data = 1.0 - combined_dist.flatten()  # 转换距离为相似度
                
                connectivities = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                
                # 创建整合的AnnData对象
                print("创建整合的AnnData对象...")
                integrated = rna_subset.copy()
                
                # 存储整合后的连接矩阵
                integrated.obsp['connectivities'] = connectivities
                
                # 添加ATAC信息
                integrated.uns['atac'] = atac_subset
                
                # 提前计算UMAP embedding以优化后续步骤
                print("计算UMAP嵌入...")
                try:
                    sc.tl.umap(integrated)
                except Exception as e:
                    print(f"UMAP计算失败: {e}，跳过UMAP计算")
                
                elapsed_time = time.time() - start_time
                print(f"WNN整合完成，耗时: {elapsed_time:.2f}秒")
                return integrated
            except Exception as e:
                print(f"WNN整合失败: {e}")
                print(traceback.format_exc())
                print("回退到Concatenate方法...")
                integration_method = 'Concatenate'
        
        # 处理方法: 简单连接
        if integration_method == 'Concatenate':
            print("使用特征连接方法整合数据")
            
            try:
                # 创建整合的AnnData对象
                print("连接PCA和LSI特征...")
                atac_rep = atac_subset.obsm[atac_rep_key]
                
                integrated = rna_subset.copy()
                
                # 连接RNA PCA和ATAC LSI特征
                integrated.obsm['X_integrated'] = np.concatenate([
                    rna_subset.obsm['X_pca'],
                    atac_rep
                ], axis=1)
                
                # 存储原始数据引用
                integrated.uns['atac'] = atac_subset
                
                # 计算整合后的邻域图
                print("计算整合的邻域图...")
                sc.pp.neighbors(integrated, n_neighbors=30, use_rep='X_integrated')
                
                # 计算UMAP
                print("计算UMAP嵌入...")
                try:
                    sc.tl.umap(integrated)
                except Exception as e:
                    print(f"UMAP计算失败: {e}，跳过UMAP计算")
                
                elapsed_time = time.time() - start_time
                print(f"Concatenate整合完成，耗时: {elapsed_time:.2f}秒")
                return integrated
            except Exception as e:
                print(f"Concatenate整合失败: {e}")
                print(traceback.format_exc())
                print("回退到简单数据合并...")
                
                # 紧急回退 - 简单复制RNA数据并保存ATAC信息
                integrated = rna_subset.copy()
                integrated.uns['atac'] = atac_subset
                
                # 如果有可用的降维结果，则使用
                try:
                    sc.pp.neighbors(integrated, n_neighbors=30)
                except:
                    print("无法计算邻域图，继续处理...")
                
                elapsed_time = time.time() - start_time
                print(f"简单数据合并完成，耗时: {elapsed_time:.2f}秒")
                return integrated
        
        # 处理方法: Mutual Nearest Neighbors (MNN)
        elif integration_method == 'MNN':
            print("使用MNN方法整合数据")
            # MNN是一种相对复杂的方法，此处实现简化版本
            
            try:
                # 获取降维特征
                rna_pca = rna_subset.obsm['X_pca']
                atac_rep = atac_subset.obsm[atac_rep_key]
                
                # 确保维度一致
                min_dims = min(rna_pca.shape[1], atac_rep.shape[1])
                rna_pca = rna_pca[:, :min_dims]
                atac_rep = atac_rep[:, :min_dims]
                
                print(f"MNN整合使用 {min_dims} 个维度")
                
                # 找到互最近邻居
                print("计算RNA到ATAC的KNN...")
                rna_to_atac = NearestNeighbors(n_neighbors=3).fit(atac_rep)
                _, rna_to_atac_idx = rna_to_atac.kneighbors(rna_pca)
                
                print("计算ATAC到RNA的KNN...")
                atac_to_rna = NearestNeighbors(n_neighbors=3).fit(rna_pca)
                _, atac_to_rna_idx = atac_to_rna.kneighbors(atac_rep)
                
                # 找到互相都是最近邻的对
                print("找到互最近邻对...")
                mnn_pairs = []
                for i in range(len(rna_pca)):
                    for j in rna_to_atac_idx[i]:
                        # 检查j是否有i作为它的邻居
                        if i in atac_to_rna_idx[j]:
                            mnn_pairs.append((i, j))
                
                print(f"找到 {len(mnn_pairs)} 个互最近邻对")
                
                if len(mnn_pairs) == 0:
                    print("没有找到互最近邻对，回退到Concatenate方法")
                    return integrate_rna_atac(rna, atac, n_pcs, use_gene_activity, 'Concatenate')
                
                # 创建整合的AnnData对象
                print("创建整合的AnnData对象...")
                integrated = rna_subset.copy()
                
                # 存储MNN对信息
                integrated.uns['mnn_pairs'] = mnn_pairs
                integrated.uns['atac'] = atac_subset
                
                # 计算并存储整合表示
                # 简单合并作为替代
                integrated.obsm['X_integrated'] = np.concatenate([rna_pca, atac_rep], axis=1)
                
                # 计算整合后的邻域图
                print("计算整合的邻域图...")
                sc.pp.neighbors(integrated, n_neighbors=30, use_rep='X_integrated')
                
                # 计算UMAP
                print("计算UMAP嵌入...")
                try:
                    sc.tl.umap(integrated)
                except Exception as e:
                    print(f"UMAP计算失败: {e}，跳过UMAP计算")
                
                elapsed_time = time.time() - start_time
                print(f"MNN整合完成，耗时: {elapsed_time:.2f}秒")
                return integrated
            except Exception as e:
                print(f"MNN整合失败: {e}")
                print(traceback.format_exc())
                print("回退到Concatenate方法...")
                return integrate_rna_atac(rna, atac, n_pcs, use_gene_activity, 'Concatenate')
        
        # 处理方法: Harmony
        elif integration_method == 'Harmony':
            print("使用Harmony方法整合数据")
            
            try:
                # 尝试导入harmony_pytorch
                try:
                    import harmonypy
                except ImportError:
                    print("未找到harmonypy包，回退到Concatenate方法")
                    return integrate_rna_atac(rna, atac, n_pcs, use_gene_activity, 'Concatenate')
                
                # 创建整合的AnnData对象
                print("准备Harmony整合...")
                integrated = rna_subset.copy()
                
                # 制作批次标签
                integrated.obs['data_type'] = 'RNA'
                
                # 准备输入矩阵
                X_pca = rna_subset.obsm['X_pca']
                atac_rep = atac_subset.obsm[atac_rep_key] 
                
                # 确保维度一致
                min_dims = min(X_pca.shape[1], atac_rep.shape[1])
                X_pca = X_pca[:, :min_dims]
                atac_rep = atac_rep[:, :min_dims]
                
                # 简单平均整合表示（之后会被Harmony替换）
                integrated.obsm['X_integrated'] = (X_pca + atac_rep) / 2
                
                # 保存原始数据
                integrated.uns['atac'] = atac_subset
                
                # 使用Harmony
                print("运行Harmony整合...")
                data_mat = np.concatenate([X_pca, atac_rep], axis=0)
                meta_data = np.array(['RNA'] * len(X_pca) + ['ATAC'] * len(atac_rep))
                
                # 运行Harmony
                ho = harmonypy.run_harmony(data_mat, meta_data, ['RNA', 'ATAC'])
                harmony_embeddings = ho.Z_corr.T
                
                # 只保留RNA部分
                integrated.obsm['X_harmony'] = harmony_embeddings[:len(X_pca)]
                
                # 计算邻域图
                print("计算基于Harmony的邻域图...")
                sc.pp.neighbors(integrated, use_rep='X_harmony')
                
                # 计算UMAP
                print("计算UMAP嵌入...")
                try:
                    sc.tl.umap(integrated)
                except Exception as e:
                    print(f"UMAP计算失败: {e}，跳过UMAP计算")
                
                elapsed_time = time.time() - start_time
                print(f"Harmony整合完成，耗时: {elapsed_time:.2f}秒")
                return integrated
            except Exception as e:
                print(f"Harmony整合失败: {e}")
                print(traceback.format_exc())
                print("回退到Concatenate方法...")
                return integrate_rna_atac(rna, atac, n_pcs, use_gene_activity, 'Concatenate')
        
        else:
            print(f"错误: 未知的整合方法 '{integration_method}'")
            print("使用Concatenate方法作为替代")
            return integrate_rna_atac(rna, atac, n_pcs, use_gene_activity, 'Concatenate')
    
    except Exception as e:
        print(f"整合过程中发生错误: {e}")
        print(traceback.format_exc())
        
        # 在错误情况下，返回简单复制的RNA数据
        print("出错情况下，返回简单的RNA数据")
        integrated = rna_subset.copy()
        return integrated 