# Integrated Multi-omics Data Processing Script

这个脚本整合了多组学数据处理的功能，可以将 multiome 数据和单细胞数据转换为基因视角的 NPZ 格式，并进行 K 折交叉验证划分。脚本包含了多项改进，包括使用真实启动子序列、增强 ATAC 信号密度、基因 ID 转换和信号强度调整等功能。

## 主要功能

1. **数据处理**
   - 处理 multiome 数据（RNA + ATAC）
   - 处理单细胞数据（RNA + 借用 ATAC）
   - 支持多组织数据处理

2. **数据改进**
   - 使用真实启动子序列替换随机序列
   - 增强 ATAC 信号密度（目标密度 86.3%）
   - 将基因符号转换为 ENSEMBL ID
   - 调整信号强度匹配参考数据分布

3. **数据格式**
   - 输入：h5ad 格式的多组学数据
   - 输出：NPZ 格式的基因视角数据
   - 包含：samples（基因 ID）、rna（表达量）、atac（染色质可及性）、sequence（DNA 序列）

4. **交叉验证**
   - 支持 K 折交叉验证（默认 5 折）
   - 自动划分训练集和测试集
   - 保持数据分布一致性

## 依赖项

```bash
numpy
scanpy
anndata
scikit-learn
scipy
matplotlib
biopython
```

## 使用方法

### 基本用法

```bash
python integrated_convert_to_cv_gene_view.py [参数]
```

### 命令行参数

#### 基本参数
- `--output`: 输出目录（默认：'data/improved_gene_view_datasets'）
- `--seq-length`: 序列长度（默认：2000）
- `--n-folds`: 交叉验证折数（默认：5）
- `--random-state`: 随机种子（默认：42）

#### 数据参数
- `--tissues`: 要处理的 multiome 组织列表（默认：['pbmc', 'brain', 'jejunum']）
- `--sc-tissue`: 单细胞组织名称（默认：'pbmc'）
- `--multiome-tissue`: multiome 组织名称，用于借用 ATAC 信号（默认：'pbmc'）

#### 文件路径参数
- `--promoter-file`: 启动子序列 FASTA 文件（默认：'data/processed/promoter_sequences.fa'）
- `--promoter-info`: 启动子信息文件（默认：'data/processed/promoter_info.tsv'）
- `--gene-mapping`: 基因符号到 ENSEMBL ID 的映射文件（默认：'data/processed/gene_symbol_to_ensembl.tsv'）
- `--reference-file`: 参考数据集文件，用于信号强度调整（默认：'data/reference/reference_atac_data.npz'）

#### 功能开关
- `--enhance-atac`: 增强 ATAC 信号密度（默认：True）
- `--adjust-signal`: 调整 ATAC 信号强度（默认：True）
- `--adjust-method`: 信号强度调整方法（'linear' 或 'distribution'，默认：'distribution'）
- `--process-sc`: 是否处理单细胞数据（默认：False）

### 使用示例

1. **只处理 multiome 数据**
```bash
python integrated_convert_to_cv_gene_view.py --tissues pbmc brain jejunum
```

2. **只处理单细胞数据**
```bash
python integrated_convert_to_cv_gene_view.py --process-sc --sc-tissue pbmc --multiome-tissue pbmc
```

3. **同时处理两种数据**
```bash
python integrated_convert_to_cv_gene_view.py --tissues pbmc brain jejunum --process-sc --sc-tissue pbmc --multiome-tissue pbmc
```

4. **自定义参数处理**
```bash
python integrated_convert_to_cv_gene_view.py \
    --output data/custom_output \
    --seq-length 3000 \
    --n-folds 10 \
    --tissues pbmc brain \
    --process-sc \
    --sc-tissue pbmc \
    --multiome-tissue pbmc \
    --promoter-file data/custom/promoters.fa \
    --gene-mapping data/custom/gene_map.tsv \
    --enhance-atac \
    --adjust-signal \
    --adjust-method distribution
```

## 输出文件结构

```
output_directory/
├── pbmc/
│   ├── fold_0_train.npz
│   ├── fold_0_test.npz
│   ├── fold_1_train.npz
│   ├── fold_1_test.npz
│   └── ...
├── brain/
│   ├── fold_0_train.npz
│   ├── fold_0_test.npz
│   └── ...
├── jejunum/
│   ├── fold_0_train.npz
│   ├── fold_0_test.npz
│   └── ...
└── pbmc_sc/
    ├── fold_0_train.npz
    ├── fold_0_test.npz
    └── ...
```

## NPZ 文件内容

每个 NPZ 文件包含以下数据：
- `samples`: 基因 ID 数组
- `rna`: RNA 表达数据矩阵
- `atac`: ATAC 信号数据矩阵
- `sequence`: DNA 序列数据矩阵

## 注意事项

1. 确保所有输入文件路径正确
2. 处理大量数据时注意内存使用
3. 建议先使用小数据集测试参数设置
4. 单细胞数据处理需要对应的 multiome 数据用于 ATAC 信号借用

## 错误处理

脚本包含以下错误处理机制：
- 文件路径检查
- 数据格式验证
- 内存使用监控
- 进度报告和日志输出

## 性能优化

- 使用 numpy 向量化操作
- 支持稀疏矩阵处理
- 分批处理大型数据集
- 内存使用优化

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进代码。提交时请：
1. 清晰描述问题或改进
2. 提供测试用例
3. 更新文档
4. 遵循代码风格指南

## 许可证

MIT License 