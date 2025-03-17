 # 多组学数据处理流程 - 基于加权最近邻整合方法(WNN)

## 1. 项目概述

本项目实现了一套完整的单细胞多组学数据分析流程，用于处理和整合ATAC-seq与RNA-seq数据。主要使用加权最近邻(Weighted Nearest Neighbors, WNN)方法进行多模态数据整合，实现对细胞染色质可及性和基因表达谱的联合分析。流程支持多种组织类型(PBMC、脑组织、空肠组织)和单细胞ADT数据处理。

## 2. 数据集说明

本流程处理以下数据类型：
- **多组学数据(Multiome)**：10x Genomics同细胞ATAC+RNA数据
  - PBMC (外周血单核细胞)
  - 脑组织 (Brain)
  - 空肠组织 (Jejunum)
- **单细胞数据**：10x Genomics单细胞RNA+ADT(抗体标记)数据

## 3. 环境配置与依赖

### 3.1 主要依赖库
```
scanpy>=1.8.0      # 单细胞数据分析
anndata>=0.8.0     # 单细胞数据结构
pyranges>=0.0.117  # 基因组区间操作
pysam>=0.19.0      # BAM/SAM/CRAM格式处理
numpy>=1.20.0      # 数值计算
pandas>=1.3.0      # 数据处理
matplotlib>=3.4.0  # 可视化
scipy>=1.7.0       # 科学计算
protobuf>=3.20.0   # 数据序列化
psutil>=5.9.0      # 系统资源监控
```

### 3.2 系统要求
- 推荐内存：≥32GB (处理大型数据集时≥64GB)
- 存储空间：≥100GB
- CPU：≥8核心

## 4. 数据处理流程详述

### 4.1 数据加载

#### 4.1.1 多组学数据加载
```python
# 从10x h5文件加载RNA和ATAC数据
adata = sc.read_10x_h5(h5_file)
# 分离RNA表达数据
rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
# 分离ATAC峰数据
atac = adata[:, adata.var['feature_types'] == 'Peaks'].copy()
```

#### 4.1.2 片段文件索引
为加速峰信号检索，需创建ATAC fragments文件的tabix索引：
```bash
# 确保排序
bgzip -c fragments.tsv > fragments.tsv.gz
# 创建索引
tabix -p bed fragments.tsv.gz
```

### 4.2 RNA数据预处理

#### 4.2.1 质控与过滤
```python
# 计算每个细胞的基因数和总计数
sc.pp.calculate_qc_metrics(rna, inplace=True)
# 过滤低质量细胞
rna = rna[rna.obs.n_genes_by_counts > 200].copy()
rna = rna[rna.obs.total_counts < 25000].copy()
# 过滤线粒体基因比例高的细胞
rna = rna[rna.obs.pct_counts_mt < 20].copy()
```

#### 4.2.2 归一化与转换
```python
# 总计数归一化并对数转换
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
# 识别高变异基因
sc.pp.highly_variable_genes(rna, n_top_genes=2000)
# 使用高变异基因子集
rna_hvg = rna[:, rna.var.highly_variable].copy()
```

#### 4.2.3 降维
```python
# 主成分分析(PCA)
sc.pp.scale(rna_hvg)
sc.tl.pca(rna_hvg, n_comps=50, svd_solver='arpack')
# 将PCA结果转移回完整数据集
rna.obsm['X_pca'] = rna_hvg.obsm['X_pca']
```

### 4.3 ATAC数据预处理

#### 4.3.1 质控与过滤
```python
# 计算每个细胞的峰数和总计数
atac_counts_per_cell = atac.X.sum(axis=1)
atac.obs['n_peaks'] = atac_counts_per_cell
# 过滤低质量细胞和峰
atac = atac[atac.obs.n_peaks > 100].copy()
# 过滤低信号峰
min_cells = 5  # 至少5个细胞表达的峰
atac = atac[:, atac.X.sum(axis=0) >= min_cells].copy()
```

#### 4.3.2 TF-IDF变换
对ATAC数据应用TF-IDF变换，增强重要峰信号并减少批次效应：
```python
# 计算项频率-逆文档频率(TF-IDF)变换
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(norm='l2', use_idf=True)
atac_tfidf = transformer.fit_transform(atac.X)
```

#### 4.3.3 潜在语义索引(LSI)降维
```python
# 使用截断SVD计算LSI
from sklearn.decomposition import TruncatedSVD
n_comps = 50
svd = TruncatedSVD(n_components=n_comps, random_state=42)
X_lsi = svd.fit_transform(atac_tfidf)
# 存储LSI结果，去除第一个组件(捕获技术噪声)
atac.obsm['X_lsi'] = X_lsi
atac.obsm['X_lsi_filtered'] = X_lsi[:, 1:]
```

### 4.4 峰区注释与基因活性矩阵

#### 4.4.1 峰到基因注释
将ATAC峰注释到最近的基因，使用两种方式：
1. 基于基因组注释文件
   ```python
   # 解析GTF文件获取基因位置
   import pyranges as pr
   genes = pr.read_gtf(genome_annotation, as_df=True)
   gene_coords = genes[genes['Feature'] == 'gene']
   ```

2. 或使用峰注释文件
   ```python
   # 从注释文件中读取峰区对应基因
   peak_gene_map = pd.read_csv(peak_annotation_file, sep='\t')
   ```

#### 4.4.2 基因活性矩阵计算
基因活性矩阵将染色质可及性信号映射到基因，对WNN整合至关重要：
```python
# 创建稀疏矩阵记录每个基因的活跃度
from scipy import sparse
import numpy as np

# 根据峰区距离基因的远近赋予不同权重
# TSS区域±2kb权重为1.0
# 2-10kb内权重为0.5
# 10-100kb内权重为0.2

gene_activity = sparse.lil_matrix((atac.shape[0], len(gene_list)))

# 遍历每个峰区
for peak_idx, peak_name in enumerate(atac.var_names):
    chrom, start_end = peak_name.split(':')
    start, end = map(int, start_end.split('-'))
    peak_center = (start + end) // 2
    
    # 获取峰区影响的基因
    for gene_idx, gene_info in enumerate(gene_coords):
        gene_tss = gene_info['start'] if gene_info['strand'] == '+' else gene_info['end']
        distance = abs(peak_center - gene_tss)
        
        # 根据距离分配权重
        if distance <= 2000:
            weight = 1.0
        elif distance <= 10000:
            weight = 0.5
        elif distance <= 100000:
            weight = 0.2
        else:
            continue
            
        # 累加峰区对基因的贡献
        gene_idx = gene_list.index(gene_info['gene_name'])
        peak_cells = atac.X[:, peak_idx].nonzero()[0]
        gene_activity[peak_cells, gene_idx] += weight
```

### 4.5 WNN整合方法详解

加权最近邻(WNN)方法通过同时考虑RNA和ATAC数据来构建细胞关系图，是实现多模态整合的核心：

#### 4.5.1 构建模态特定KNN图
```python
from sklearn.neighbors import NearestNeighbors

# RNA模态KNN图
n_neighbors = 20
rna_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(rna.obsm['X_pca'])
rna_dist, rna_indices = rna_neighbors.kneighbors()

# ATAC模态KNN图(使用基因活性矩阵的LSI表示)
atac_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(gene_activity.obsm['X_lsi_filtered'])
atac_dist, atac_indices = atac_neighbors.kneighbors()
```

#### 4.5.2 距离归一化
```python
# 归一化距离矩阵到[0,1]范围
rna_dist = rna_dist / rna_dist.max()
atac_dist = atac_dist / atac_dist.max()
```

#### 4.5.3 WNN加权图构建
WNN特殊之处在于对两种模态赋予平衡权重，避免任一模态主导整合结果：
```python
# 模态权重参数(RNA和ATAC各占一半权重)
alpha = 0.5

# 计算加权距离
combined_dist = alpha * rna_dist + (1 - alpha) * atac_dist

# 构建加权连接图
n_cells = len(common_cells)
rows = np.repeat(np.arange(n_cells), n_neighbors)
cols = rna_indices.flatten()
# 将距离转为相似度
weights = 1.0 - combined_dist.flatten()

# 创建稀疏连接矩阵
from scipy.sparse import csr_matrix
connectivities = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))

# 将WNN图存入整合的AnnData对象
integrated = rna.copy()
integrated.obsp['connectivities'] = connectivities
```

### 4.6 聚类与降维可视化

#### 4.6.1 使用整合后的连接图进行聚类
```python
# 使用Leiden算法进行社区检测
sc.tl.leiden(integrated, resolution=0.8)

# 备选：如果Leiden失败，使用Louvain
# sc.tl.louvain(integrated, resolution=0.8)
```

#### 4.6.2 UMAP降维可视化
```python
# 计算UMAP嵌入
sc.tl.umap(integrated)

# 可视化聚类结果
sc.pl.umap(integrated, color='leiden', legend_loc='on data')
```

### 4.7 细胞类型注释

#### 4.7.1 差异表达分析
```python
# 计算差异表达基因
sc.tl.rank_genes_groups(integrated, 'leiden', method='wilcoxon')

# 提取每个簇的标记基因
marker_genes = pd.DataFrame(integrated.uns['rank_genes_groups']['names'])
```

#### 4.7.2 基于标记基因注释细胞类型
```python
# 定义组织特异的标记基因
cell_type_markers = {
    'pbmc': {
        'B cell': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
        'CD14+ monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9'],
        'CD4 T cell': ['CD3D', 'CD3E', 'CD4', 'IL7R'],
        'CD8 T cell': ['CD3D', 'CD3E', 'CD8A', 'CD8B'],
        'NK cell': ['GNLY', 'NKG7', 'KLRD1', 'NCAM1'],
        # 其他细胞类型
    }
}

# 为每个簇分配细胞类型
cluster_to_celltype = {}
for cluster in integrated.obs['leiden'].unique():
    # 获取簇特异的标记基因
    cluster_markers = marker_genes[cluster].head(50).tolist()
    
    # 与已知标记基因匹配
    max_overlap = 0
    assigned_type = 'Unknown'
    
    for cell_type, markers in cell_type_markers['pbmc'].items():
        overlap = len(set(cluster_markers).intersection(markers))
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_type = cell_type
    
    cluster_to_celltype[cluster] = assigned_type

# 添加细胞类型注释
integrated.obs['cell_type'] = integrated.obs['leiden'].map(cluster_to_celltype)
```

## 5. 输出文件说明

### 5.1 中间检查点文件
- `{tissue}_integrated_YYYYMMDD_HHMMSS.h5ad`: 处理后的AnnData对象

### 5.2 分析结果文件
- `{tissue}_multiome_atac_improved_v2.h5ad`: 最终整合后的多组学数据
- `{tissue}_atac_improved_v2_cell_counts.csv`: 细胞类型计数
- `cell_type_summary.csv`: 所有数据集细胞类型统计
- `cell_type_heatmap.png`: 细胞类型分布热图
- `cell_type_barplot.png`: 细胞类型计数条形图

## 6. 内存优化策略

对于大型数据集，流程采用以下优化策略：
1. 分块处理基因活性矩阵(每次1000个基因)
2. 稀疏矩阵存储(CSR格式)
3. 内存使用监控，超过阈值(默认80%)时采用低内存模式
4. 使用检查点保存中间结果

## 7. 参考文献

1. Stuart T, Butler A, et al. (2019). "Comprehensive Integration of Single-Cell Data." Cell, 177(7), 1888-1902.
2. Hao Y, Hao S, et al. (2021). "Integrated analysis of multimodal single-cell data." Cell, 184(13), 3573-3587.
3. Wolf FA, Angerer P, Theis FJ (2018). "SCANPY: large-scale single-cell gene expression data analysis." Genome Biology, 19:15.
4. Pliner HA, Packer JS, et al. (2018). "Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data." Molecular Cell, 71(5), 858-871.
5. McInnes L, Healy J, Melville J (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426.