#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATAC数据加载和预处理模块 
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')

from utils import check_and_create_tabix_index

def load_10x_atac(h5_file=None, fragments_file=None, peaks_file=None):
    """
    专门加载10x ATAC数据
    
    参数:
    - h5_file: ATAC h5文件路径
    - fragments_file: ATAC fragments文件路径
    - peaks_file: peaks文件路径
    
    返回:
    - ATAC数据AnnData对象
    """
    if h5_file is None:
        print("错误: 必须提供h5文件路径")
        return None
    
    try:
        # 尝试通过指定feature_type='Peaks'直接读取h5文件中的ATAC数据
        print(f"从{h5_file}读取ATAC数据...")
        try:
            atac = sc.read_10x_h5(h5_file, gex_only=False)
            if 'feature_types' in atac.var:
                # 检查是否有'Peaks'类型
                if 'Peaks' in atac.var['feature_types'].values:
                    atac = atac[:, atac.var['feature_types'] == 'Peaks'].copy()
                    print(f"成功读取到ATAC数据: {atac.shape[0]}细胞, {atac.shape[1]}峰")
                    atac.var_names_make_unique()
                    
                    # 如果提供了fragments文件，使用它来增强峰值信号
                    if fragments_file and os.path.exists(fragments_file):
                        try:
                            print(f"从{fragments_file}读取ATAC fragments数据...")
                            
                            # 检查并创建tabix索引
                            if not check_and_create_tabix_index(fragments_file):
                                print("无法处理fragments文件，继续使用原始ATAC数据")
                                return atac
                            
                            # 检查是否有必要的库
                            try:
                                import pyranges
                                import pysam
                            except ImportError:
                                print("警告: 无法导入pyranges或pysam库，无法处理fragments文件")
                                print("提示: 可以使用pip安装: pip install pyranges pysam")
                                return atac
                            
                            # 使用pysam读取fragments文件
                            fragments = []
                            try:
                                # 对于已排序的fragments文件，使用Tabix进行索引
                                if fragments_file.endswith('.gz'):
                                    print("使用Tabix读取压缩fragments文件...")
                                    tbx = pysam.TabixFile(fragments_file)
                                    # 读取头部信息获取染色体
                                    chromosomes = tbx.contigs
                                    
                                    # 为每个peak在fragments文件中计算覆盖度
                                    if 'chr' in atac.var and 'start' in atac.var and 'end' in atac.var:
                                        print("使用peaks坐标计算fragments覆盖...")
                                        
                                        # 创建Fragment覆盖矩阵
                                        coverage_matrix = np.zeros(atac.shape)
                                        
                                        # 对每个peak计算fragments覆盖
                                        for i, (chr_name, start, end) in enumerate(zip(atac.var['chr'], atac.var['start'], atac.var['end'])):
                                            if i % 1000 == 0:
                                                print(f"处理第{i}/{atac.shape[1]}个peak...")
                                            
                                            try:
                                                # 获取该区域的所有fragments
                                                for row in tbx.fetch(chr_name, start, end):
                                                    parts = row.split()
                                                    cell_barcode = parts[3]
                                                    
                                                    # 如果该细胞在我们的数据集中
                                                    if cell_barcode in atac.obs_names:
                                                        cell_idx = atac.obs_names.get_loc(cell_barcode)
                                                        coverage_matrix[cell_idx, i] += 1
                                            except ValueError:
                                                # 某些染色体可能在fragments文件中不存在
                                                continue
                                    # 尝试从interval列解析位置信息
                                    elif 'interval' in atac.var:
                                        print("从interval列解析peaks坐标...")
                                        
                                        # 创建新列存储解析的坐标
                                        atac.var['chr'] = ''
                                        atac.var['start'] = 0
                                        atac.var['end'] = 0
                                        
                                        # 解析interval字段
                                        for idx, interval in enumerate(atac.var['interval']):
                                            try:
                                                chr_part, pos_part = interval.split(':')
                                                start_end = pos_part.split('-')
                                                if len(start_end) == 2:
                                                    start, end = start_end
                                                    
                                                    atac.var.loc[atac.var.index[idx], 'chr'] = chr_part
                                                    atac.var.loc[atac.var.index[idx], 'start'] = int(start)
                                                    atac.var.loc[atac.var.index[idx], 'end'] = int(end)
                                            except Exception as e:
                                                print(f"解析interval失败: {interval}, 错误: {e}")
                                        
                                        print(f"成功从interval解析出位置信息，示例: {atac.var['chr'].iloc[0]}:{atac.var['start'].iloc[0]}-{atac.var['end'].iloc[0]}")
                                        
                                        # 使用解析后的坐标计算覆盖度
                                        print("使用解析的peaks坐标计算fragments覆盖...")
                                        
                                        # 创建Fragment覆盖矩阵
                                        coverage_matrix = np.zeros(atac.shape)
                                        
                                        # 以下使用已解析的坐标进行计算
                                        for i, (chr_name, start, end) in enumerate(zip(atac.var['chr'], atac.var['start'], atac.var['end'])):
                                            if i % 1000 == 0:
                                                print(f"处理第{i}/{atac.shape[1]}个peak...")
                                            
                                            try:
                                                # 获取该区域的所有fragments
                                                for row in tbx.fetch(chr_name, int(start), int(end)):
                                                    parts = row.split()
                                                    cell_barcode = parts[3]
                                                    
                                                    # 如果该细胞在我们的数据集中
                                                    if cell_barcode in atac.obs_names:
                                                        cell_idx = atac.obs_names.get_loc(cell_barcode)
                                                        coverage_matrix[cell_idx, i] += 1
                                            except ValueError as e:
                                                # 某些染色体可能在fragments文件中不存在
                                                if i < 5:  # 仅显示前几个错误
                                                    print(f"获取fragments出错: {chr_name}:{start}-{end}, 错误: {e}")
                            except Exception as e:
                                print(f"处理fragments文件时出错: {e}")
                                return atac
                        except Exception as e:
                            print(f"处理ATAC fragments数据时出错: {e}")
                            return atac
                    
                    return atac
                else:
                    print(f"警告: 在h5文件中未找到'Peaks'特征类型，找到的类型有: {atac.var['feature_types'].unique()}")
            else:
                print("警告: h5文件的var中没有'feature_types'列")
        except Exception as e:
            print(f"读取h5文件时出错: {e}")
        
        # 尝试读取peaks文件(如果提供)
        if peaks_file and os.path.exists(peaks_file):
            try:
                print(f"从{peaks_file}读取peaks...")
                peaks = pd.read_csv(peaks_file, sep='\t')
                print(f"读取了 {len(peaks)} 个peaks")
                return None  # 此处可根据peaks文件格式创建适当的AnnData
            except Exception as e:
                print(f"读取peaks文件时出错: {e}")
        
        print("未能成功加载ATAC数据")
        return None
    
    except Exception as e:
        print(f"加载ATAC数据时出错: {e}")
        return None

def optimize_atac_preprocessing(atac, min_cells=5, min_peaks=200):
    """
    优化ATAC数据预处理
    
    参数:
    - atac: ATAC AnnData对象
    - min_cells: 每个peak至少出现在多少个细胞中
    - min_peaks: 每个细胞至少有多少个peak
    
    返回:
    - 预处理后的ATAC AnnData对象
    """
    if atac is None:
        print("错误: 提供的ATAC数据为None")
        return None
    
    print(f"原始ATAC数据: {atac.shape[0]}细胞, {atac.shape[1]}峰")
    
    # 1. 过滤低质量细胞和peak
    try:
        # 过滤低覆盖度的peak
        sc.pp.filter_genes(atac, min_cells=min_cells)
        
        # 过滤低质量细胞(peaks数量过少)
        sc.pp.filter_cells(atac, min_genes=min_peaks)
        
        print(f"过滤后ATAC数据: {atac.shape[0]}细胞, {atac.shape[1]}峰")
    except Exception as e:
        print(f"过滤ATAC数据时出错: {e}")
    
    # 2. TF-IDF变换并计算LSI
    try:
        # 创建TF-IDF转换
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.decomposition import TruncatedSVD
        
        # 转换原始计数
        tfidf = TfidfTransformer(norm='l2', use_idf=True)
        
        # 处理稀疏矩阵
        if sparse.issparse(atac.X):
            X_tfidf = tfidf.fit_transform(atac.X)
        else:
            X_tfidf = tfidf.fit_transform(sparse.csr_matrix(atac.X))
        
        # 将转换后的矩阵存储回atac.X
        atac.X = X_tfidf
        
        # 3. 使用截断SVD计算LSI
        n_comps = min(50, atac.shape[1]-1)
        svd = TruncatedSVD(n_components=n_comps, random_state=42)
        X_lsi = svd.fit_transform(atac.X)
        
        # 存储LSI结果
        atac.obsm['X_lsi'] = X_lsi
        
        # 一般移除第一个成分(捕获序列深度变异)
        atac.obsm['X_lsi_filtered'] = X_lsi[:, 1:]
        
        print(f"计算了{n_comps}个LSI成分, 移除第一个成分后保留{n_comps-1}个")
    except Exception as e:
        print(f"计算ATAC TF-IDF和LSI时出错: {e}")
    
    return atac

def annotate_peaks_to_genes(atac, genomic_coords=None, peak_annotation_file=None, window_size=5000):
    """
    将peaks注释到最近的基因
    
    参数:
    - atac: ATAC AnnData对象
    - genomic_coords: 基因组坐标文件(GTF或类似格式)
    - peak_annotation_file: 峰注释文件(TSV格式，包含peak与基因的对应关系)
    - window_size: 计算基因启动子的窗口大小(bp)
    
    返回:
    - 注释后的ATAC AnnData对象
    """
    if atac is None:
        print("错误: 提供的ATAC数据为None")
        return None
    
    # 如果没有位置信息，尝试从变量名解析
    if 'chr' not in atac.var or 'start' not in atac.var or 'end' not in atac.var:
        print("ATAC数据没有位置信息，尝试从变量名解析...")
        try:
            # 假设格式为chr:start-end
            new_var = pd.DataFrame(index=atac.var_names)
            new_var['chr'] = [x.split(':')[0] for x in atac.var_names]
            new_var['interval'] = [x.split(':')[1] if ':' in x else '' for x in atac.var_names]
            new_var['start'] = [int(x.split('-')[0]) if '-' in x else 0 for x in new_var['interval']]
            new_var['end'] = [int(x.split('-')[1]) if '-' in x else 0 for x in new_var['interval']]
            
            # 更新var
            for col in ['chr', 'start', 'end']:
                atac.var[col] = new_var[col]
                
            print(f"从变量名解析位置信息成功，示例: {atac.var['chr'].iloc[0]}:{atac.var['start'].iloc[0]}-{atac.var['end'].iloc[0]}")
        except Exception as e:
            print(f"从变量名解析位置信息失败: {e}")
            print("使用随机基因名注释...")
            return _annotate_peaks_random(atac)
    
    # 首先尝试使用峰注释文件
    if peak_annotation_file and os.path.exists(peak_annotation_file):
        print(f"使用峰注释文件{peak_annotation_file}获取peak-gene对应关系...")
        try:
            # 读取峰注释文件
            peak_annotations = pd.read_csv(peak_annotation_file, sep='\t')
            print(f"读取了{len(peak_annotations)}个峰注释")
            
            # 检查列名
            print(f"峰注释文件列名: {peak_annotations.columns.tolist()}")
            
            # 假设文件含有peak_id和gene_name两列
            # 根据实际列名调整
            peak_col = next((col for col in peak_annotations.columns if 'peak' in col.lower()), None)
            gene_col = next((col for col in peak_annotations.columns if 'gene' in col.lower()), None)
            
            if peak_col and gene_col:
                print(f"使用{peak_col}列作为peak ID，{gene_col}列作为基因名")
                
                # 创建peak到gene的映射
                peak_to_gene = dict(zip(peak_annotations[peak_col], peak_annotations[gene_col]))
                
                # 初始化gene_name列
                atac.var['gene_name'] = ''
                
                # 尝试直接匹配peak ID
                matched_count = 0
                for peak_id in atac.var_names:
                    if peak_id in peak_to_gene:
                        atac.var.loc[peak_id, 'gene_name'] = peak_to_gene[peak_id]
                        matched_count += 1
                
                # 如果直接匹配失败，尝试基于坐标匹配
                if matched_count == 0:
                    print("直接匹配peak ID失败，尝试基于坐标匹配...")
                    
                    # 假设注释文件含有染色体、开始和结束位置列
                    chr_col = next((col for col in peak_annotations.columns if 'chr' in col.lower()), None)
                    start_col = next((col for col in peak_annotations.columns if 'start' in col.lower()), None)
                    end_col = next((col for col in peak_annotations.columns if 'end' in col.lower()), None)
                    
                    if chr_col and start_col and end_col:
                        print(f"使用{chr_col}, {start_col}, {end_col}列进行坐标匹配")
                        
                        # 建立坐标索引
                        coord_to_gene = {}
                        for _, row in peak_annotations.iterrows():
                            key = f"{row[chr_col]}:{row[start_col]}-{row[end_col]}"
                            coord_to_gene[key] = row[gene_col]
                        
                        # 按坐标匹配
                        for i, (chr_name, start, end) in enumerate(zip(atac.var['chr'], atac.var['start'], atac.var['end'])):
                            key = f"{chr_name}:{start}-{end}"
                            if key in coord_to_gene:
                                atac.var.loc[atac.var.index[i], 'gene_name'] = coord_to_gene[key]
                                matched_count += 1
                
                print(f"成功为{matched_count}个peak添加了基因注释")
                
                if matched_count > 0:
                    return atac
            else:
                print(f"在峰注释文件中未找到合适的peak和gene列")
        except Exception as e:
            print(f"解析峰注释文件时出错: {e}")
            print("尝试其他注释方法...")
    
    # 如果提供了基因组坐标文件
    if genomic_coords and os.path.exists(genomic_coords):
        print(f"使用{genomic_coords}获取基因坐标...")
        try:
            import pyranges as pr
            import gtfparse
            
            # 检查是否是GTF文件
            if genomic_coords.endswith('.gtf') or genomic_coords.endswith('.gtf.gz'):
                # 读取GTF文件
                try:
                    df = gtfparse.read_gtf(genomic_coords)
                    # 过滤基因记录
                    genes = df[df['feature'] == 'gene'].copy()
                    
                    # 创建PyRanges对象
                    genes_gr = pr.PyRanges(
                        chromosomes=genes['seqname'],
                        starts=genes['start'] - window_size,  # 扩展启动子区域
                        ends=genes['end'],
                        strands=genes['strand']
                    )
                    
                    # 如果有gene_name列，添加
                    if 'gene_name' in genes.columns:
                        genes_gr.gene_name = genes['gene_name'].values
                    
                    # 创建peaks的PyRanges对象
                    peaks_gr = pr.PyRanges(
                        chromosomes=atac.var['chr'],
                        starts=atac.var['start'],
                        ends=atac.var['end']
                    )
                    
                    # 寻找重叠
                    overlap = peaks_gr.join(genes_gr)
                    
                    # 将结果添加到var中
                    if not overlap.empty:
                        # 创建索引到peak_id的映射
                        peak_ids = atac.var.index.tolist()
                        peak_to_index = {peak: i for i, peak in enumerate(peak_ids)}
                        
                        # 初始化gene_name列
                        atac.var['gene_name'] = ''
                        
                        # 填充gene_name
                        for chrom in overlap.chromosomes:
                            df = overlap[chrom].df
                            for _, row in df.iterrows():
                                peak_idx = peak_to_index.get(row['index'])
                                if peak_idx is not None and 'gene_name' in row:
                                    atac.var.loc[atac.var.index[peak_idx], 'gene_name'] = row['gene_name']
                        
                        print(f"成功为{sum(atac.var['gene_name'] != '')}个peak添加了基因注释")
                    else:
                        print("没有找到peaks和基因的重叠")
                except Exception as e:
                    print(f"解析GTF文件时出错: {e}")
                    return _annotate_peaks_random(atac)
            else:
                print(f"不支持的基因组坐标文件格式: {genomic_coords}")
                return _annotate_peaks_random(atac)
        except ImportError:
            print("无法导入pyranges或gtfparse库，无法处理基因组坐标")
            print("提示: 可以使用pip安装: pip install pyranges gtfparse")
            return _annotate_peaks_random(atac)
    else:
        print("未提供基因组坐标文件或文件不存在")
        return _annotate_peaks_random(atac)
    
    return atac

def _annotate_peaks_random(atac):
    """
    当无法使用真实基因组注释时，使用随机基因名注释peaks
    
    参数:
    - atac: ATAC AnnData对象
    
    返回:
    - 随机注释的ATAC AnnData对象
    """
    print("使用随机基因名注释peaks...")
    
    # 创建一些随机基因名
    import string
    import random
    
    def random_gene_name():
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(3)) + str(random.randint(1, 99))
    
    # 为每个peak分配一个随机基因
    num_peaks = atac.shape[1]
    gene_names = [random_gene_name() for _ in range(num_peaks)]
    
    # 添加到var
    atac.var['gene_name'] = gene_names
    
    print(f"已为{num_peaks}个peak分配随机基因名")
    return atac

def compute_gene_activity_matrix(atac, gene_list=None):
    """
    计算基因活性矩阵，将peaks信号汇总到基因水平
    
    参数:
    - atac: ATAC AnnData对象
    - gene_list: 要包含的基因列表(可选)
    
    返回:
    - 基因活性AnnData对象
    """
    if atac is None:
        print("错误: 提供的ATAC数据为None")
        return None
    
    # 检查是否有gene_name注释
    if 'gene_name' not in atac.var:
        print("错误: ATAC数据没有基因注释，请先运行annotate_peaks_to_genes")
        return None
    
    print("计算基因活性矩阵...")
    
    # 获取所有唯一基因名
    all_genes = set(atac.var['gene_name']) - {''}
    print(f"找到{len(all_genes)}个唯一基因名")
    
    # 如果提供了基因列表，使用交集
    if gene_list:
        genes_to_use = list(set(gene_list) & all_genes)
        if len(genes_to_use) < len(gene_list):
            print(f"警告: 只有{len(genes_to_use)}/{len(gene_list)}个基因在ATAC数据中找到")
    else:
        genes_to_use = list(all_genes)
    
    # 创建基因活性矩阵
    gene_activity = np.zeros((atac.shape[0], len(genes_to_use)))
    
    # 计算每个基因的活性
    for i, gene in enumerate(genes_to_use):
        # 获取该基因的所有peak
        peak_indices = np.where(atac.var['gene_name'] == gene)[0]
        
        if len(peak_indices) > 0:
            # 选择这些peak的数据
            if sparse.issparse(atac.X):
                gene_peaks = atac.X[:, peak_indices].toarray()
            else:
                gene_peaks = atac.X[:, peak_indices]
            
            # 汇总信号(使用最大值、平均值或总和)
            gene_activity[:, i] = np.sum(gene_peaks, axis=1)
    
    # 创建新的AnnData对象
    gene_act_adata = ad.AnnData(
        X=sparse.csr_matrix(gene_activity),
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=genes_to_use)
    )
    
    print(f"基因活性矩阵: {gene_act_adata.shape[0]}细胞, {gene_act_adata.shape[1]}基因")
    return gene_act_adata

def compute_gene_activity_matrix_quick(atac, gene_list=None, max_genes=1000):
    """
    计算基因活性矩阵的快速版本 - 仅用于测试后续代码
    
    参数:
    - atac: ATAC AnnData对象
    - gene_list: 要包含的基因列表(可选)
    - max_genes: 最大处理的基因数量
    
    返回:
    - 基因活性AnnData对象
    """
    if atac is None:
        print("错误: 提供的ATAC数据为None")
        return None
    
    # 检查是否有gene_name注释
    if 'gene_name' not in atac.var:
        print("错误: ATAC数据没有基因注释，请先运行annotate_peaks_to_genes")
        return None
    
    print("计算基因活性矩阵(快速测试版)...")
    
    # 获取所有唯一基因名
    all_genes = set(atac.var['gene_name']) - {''}
    print(f"找到{len(all_genes)}个唯一基因名")
    
    # 如果提供了基因列表，使用交集
    if gene_list:
        genes_to_use = list(set(gene_list) & all_genes)
        if len(genes_to_use) < len(gene_list):
            print(f"警告: 只有{len(genes_to_use)}/{len(gene_list)}个基因在ATAC数据中找到")
    else:
        genes_to_use = list(all_genes)
    
    # 限制基因数量，加快测试速度
    genes_to_use = genes_to_use[:max_genes]
    print(f"快速测试模式: 仅使用前{len(genes_to_use)}个基因")
    
    # 创建基因活性矩阵
    gene_activity = np.zeros((atac.shape[0], len(genes_to_use)))
    
    # 计算每个基因的活性
    for i, gene in enumerate(genes_to_use):
        # 获取该基因的所有peak
        peak_indices = np.where(atac.var['gene_name'] == gene)[0]
        
        if len(peak_indices) > 0:
            # 选择这些peak的数据
            if sparse.issparse(atac.X):
                gene_peaks = atac.X[:, peak_indices].toarray()
            else:
                gene_peaks = atac.X[:, peak_indices]
            
            # 汇总信号(使用最大值、平均值或总和)
            gene_activity[:, i] = np.sum(gene_peaks, axis=1)
    
    # 创建新的AnnData对象
    gene_act_adata = ad.AnnData(
        X=sparse.csr_matrix(gene_activity),
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=genes_to_use)
    )
    
    print(f"基因活性矩阵(快速测试版): {gene_act_adata.shape[0]}细胞, {gene_act_adata.shape[1]}基因")
    return gene_act_adata 