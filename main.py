#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序 - 执行完整的ATAC+RNA分析流程
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import time
import sys
import psutil
import logging
import json
from pathlib import Path
from datetime import datetime
import traceback
from utils import check_dependencies, create_directories, check_and_create_tabix_index
from multiome_processor import process_multiome_data_improved
from singlecell_adt_processor import process_singlecell_adt_data

# 设置日志配置
def setup_logging(log_dir="logs"):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"analysis_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger('multiome_analysis')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器，同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='ATAC+RNA多组学数据分析')
    
    parser.add_argument('--tissues', type=str, default='all',
                        help='要处理的组织，用逗号分隔')
    parser.add_argument('--quick', action='store_true',
                        help='启用快速测试模式，限制基因数量以加快处理速度')
    parser.add_argument('--integration-method', type=str, default='auto',
                        help='数据整合方法: WNN, Concatenate, MNN, 或 auto (自动选择最佳方法)')
    parser.add_argument('--resolution', type=float, default=0.8,
                        help='聚类分辨率，默认 0.8')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于结果可重复性')
    parser.add_argument('--checkpoint', action='store_true',
                        help='启用检查点，保存中间分析结果')
    parser.add_argument('--resume', action='store_true',
                        help='从已保存的检查点恢复分析')
    parser.add_argument('--memory-limit', type=float, default=0.8,
                        help='最大内存使用比例 (0.0-1.0)，默认 0.8')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出信息')
    
    return parser.parse_args()

def check_file_integrity(file_path, logger):
    """检查文件完整性"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    try:
        # 检查文件大小
        size = os.path.getsize(file_path)
        if size == 0:
            logger.error(f"文件大小为0: {file_path}")
            return False
        
        # 对于h5文件，尝试验证是否可以读取
        if file_path.endswith('.h5'):
            import h5py
            try:
                with h5py.File(file_path, 'r') as f:
                    if not list(f.keys()):
                        logger.error(f"h5文件无有效数据: {file_path}")
                        return False
            except Exception as e:
                logger.error(f"无法读取h5文件 {file_path}: {e}")
                return False
        
        # 对于gzip文件，尝试读取前几行
        if file_path.endswith('.gz'):
            import gzip
            try:
                with gzip.open(file_path, 'rt') as f:
                    for _ in range(5):
                        if not next(f, None):
                            logger.error(f"gzip文件可能已损坏: {file_path}")
                            return False
            except Exception as e:
                logger.error(f"无法读取gzip文件 {file_path}: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"检查文件完整性时出错 {file_path}: {e}")
        return False

def estimate_memory_usage(file_paths, total_memory, logger):
    """估计数据处理所需内存"""
    try:
        total_file_size = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
        memory_estimate = total_file_size * 5  # 估计加载数据需要原始大小的5倍内存
        
        # 获取系统可用内存
        available_memory = psutil.virtual_memory().available
        
        logger.info(f"估计内存使用: {memory_estimate/1e9:.2f} GB")
        logger.info(f"系统可用内存: {available_memory/1e9:.2f} GB")
        
        if memory_estimate > available_memory * 0.9:
            logger.warning("警告: 估计内存使用超过可用内存的90%")
            return False
        
        return True
    except Exception as e:
        logger.error(f"估计内存使用时出错: {e}")
        return True  # 错误时假设可以继续

def save_checkpoint(data, name, output_dir, logger):
    """保存分析检查点"""
    try:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(checkpoint_dir, f"{name}_{timestamp}.h5ad")
        
        logger.info(f"保存检查点到: {filename}")
        data.write(filename)
        
        # 保存最新检查点的链接
        latest_link = os.path.join(checkpoint_dir, f"{name}_latest.h5ad")
        if os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except Exception as link_error:
                logger.warning(f"删除旧链接时出错: {link_error}")
                pass
        
        try:
            # 获取相对路径创建符号链接，避免绝对路径问题
            rel_path = os.path.basename(filename)
            os.symlink(rel_path, latest_link)
            logger.info(f"创建了符号链接: {latest_link} -> {rel_path}")
        except Exception as link_error:
            logger.warning(f"创建符号链接时出错: {link_error}")
            # 在不支持符号链接的系统上使用复制
            try:
                import shutil
                shutil.copy2(filename, latest_link)
                logger.info(f"通过复制创建了latest链接: {latest_link}")
            except Exception as copy_error:
                logger.error(f"复制文件失败: {copy_error}")
                # 错误不阻止整体流程
        
        return True
    except Exception as e:
        logger.error(f"保存检查点时出错: {e}")
        return False

def load_checkpoint(name, output_dir, logger):
    """从检查点恢复分析"""
    try:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        latest_file = os.path.join(checkpoint_dir, f"{name}_latest.h5ad")
        
        if os.path.exists(latest_file):
            logger.info(f"从检查点恢复: {latest_file}")
            return sc.read_h5ad(latest_file)
        else:
            logger.warning(f"未找到检查点: {latest_file}")
            return None
    except Exception as e:
        logger.error(f"加载检查点时出错: {e}")
        return None

def select_best_integration_method(rna, atac, gene_activity=None, logger=None):
    """
    自动选择最佳整合方法
    
    参数:
    - rna: RNA AnnData对象
    - atac: ATAC AnnData对象
    - gene_activity: 基因活性矩阵AnnData对象
    
    返回:
    - 最佳整合方法名称
    """
    if logger:
        logger.info("评估最佳整合方法...")
    
    # 判断数据规模和质量
    n_cells = rna.shape[0]
    n_genes = rna.shape[1]
    
    # 评估数据特性
    has_gene_activity = gene_activity is not None
    
    # 基于数据特性选择方法
    if n_cells < 5000:
        # 小数据集可以使用计算密集型方法
        if has_gene_activity:
            best_method = 'WNN'  # 加权最近邻提供良好整合
        else:
            best_method = 'MNN'  # 互最近邻可能更适合无基因活性的小数据集
    else:
        # 大数据集优先考虑效率
        if has_gene_activity:
            best_method = 'WNN'  # 大多情况下WNN效果最佳
        else:
            best_method = 'Concatenate'  # 简单连接效率更高
    
    if logger:
        logger.info(f"选择的最佳整合方法: {best_method}")
    
    return best_method

def analyze_tissue_data(tissue, files, marker_genes, genome_annotation, args, logger):
    """
    分析单个组织的数据
    
    参数:
    - tissue: 组织名称
    - files: 数据文件路径字典
    - marker_genes: 标记基因字典
    - genome_annotation: 基因组注释文件路径
    - args: 命令行参数
    - logger: 日志记录器
    
    返回:
    - 分析结果
    """
    logger.info(f"\n{'-'*80}")
    logger.info(f"开始分析 {tissue} 组织数据")
    logger.info(f"{'-'*80}")
    
    checkpoint_name = f"{tissue}_integrated"
    output_dir = "data/processed"
    
    # 检查是否从检查点恢复
    if args.resume:
        integrated_data = load_checkpoint(checkpoint_name, output_dir, logger)
        if integrated_data is not None:
            logger.info(f"成功从检查点恢复 {tissue} 数据分析")
            
            # 计算细胞类型计数
            cell_counts = {}
            if 'cell_type' in integrated_data.obs:
                cell_counts = integrated_data.obs['cell_type'].value_counts().to_dict()
            
            return {
                'data': integrated_data,
                'cell_counts': cell_counts
            }
    
    # 检查数据文件完整性
    if not check_file_integrity(files['h5'], logger):
        logger.error(f"h5文件完整性检查失败，跳过 {tissue} 组织分析")
        return None
    
    # 如果有fragments文件，检查完整性并创建索引
    if 'fragments' in files and files['fragments']:
        if os.path.exists(files['fragments']):
            if check_file_integrity(files['fragments'], logger):
                logger.info(f"检查并创建 {tissue} 组织的fragments文件索引...")
                check_and_create_tabix_index(files['fragments'])
            else:
                logger.warning(f"fragments文件完整性检查失败: {files['fragments']}")
                files['fragments'] = None
        else:
            logger.warning(f"fragments文件不存在: {files['fragments']}")
            files['fragments'] = None
    
    # 获取标记基因
    tissue_markers = marker_genes.get(tissue, {})
    if not tissue_markers:
        logger.warning(f"未找到 {tissue} 组织的标记基因")
    
    # 设置整合方法
    if args.integration_method == 'auto':
        # 方法将在处理过程中自动选择
        method = 'auto'
    else:
        method = args.integration_method
    
    # 设置随机种子以便结果可重现
    sc.settings.verbosity = 3 if args.verbose else 1
    sc.settings.random_seed = args.seed
    np.random.seed(args.seed)
    
    try:
        # 处理数据
        integrated_data, cell_counts = process_multiome_data_improved(
            tissue=tissue,
            h5_file=files['h5'],
            fragment_file=files.get('fragments'),
            marker_genes=tissue_markers,
            method=method,
            genome_annotation=genome_annotation,
            peak_annotation_file=files.get('peak_annotation'),
            quick_test=args.quick,
            resolution=args.resolution,
            memory_limit=args.memory_limit,
            auto_method_selection=method == 'auto'
        )
        
        # 保存检查点
        if args.checkpoint and integrated_data is not None:
            save_checkpoint(integrated_data, checkpoint_name, output_dir, logger)
        
        if integrated_data is not None:
            logger.info(f"{tissue} 组织数据分析完成")
            return {
                'data': integrated_data,
                'cell_counts': cell_counts
            }
        else:
            logger.error(f"{tissue} 组织数据分析失败")
            return None
    except MemoryError:
        logger.error(f"处理 {tissue} 组织数据时内存不足")
        # 尝试降低内存使用
        logger.info("尝试使用低内存模式重新运行...")
        try:
            integrated_data, cell_counts = process_multiome_data_improved(
                tissue=tissue,
                h5_file=files['h5'],
                fragment_file=files.get('fragments'),
                marker_genes=tissue_markers,
                method='Concatenate',  # 使用内存效率更高的方法
                genome_annotation=genome_annotation,
                peak_annotation_file=files.get('peak_annotation'),
                quick_test=True,  # 使用快速模式
                resolution=args.resolution,
                memory_limit=args.memory_limit * 0.5  # 降低内存限制
            )
            
            if integrated_data is not None:
                logger.info(f"{tissue} 组织数据分析在低内存模式下完成")
                return {
                    'data': integrated_data,
                    'cell_counts': cell_counts
                }
            else:
                logger.error(f"{tissue} 组织数据分析在低内存模式下失败")
                return None
        except Exception as e:
            logger.error(f"在低内存模式下处理 {tissue} 组织数据时出错: {e}")
            logger.error(traceback.format_exc())
            return None
    except Exception as e:
        logger.error(f"处理 {tissue} 组织数据时出错: {e}")
        logger.error(traceback.format_exc())
        return None

def main():
    """主函数：处理所有组织的多组学数据和单细胞数据"""
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("\n==== ATAC+RNA数据分析工作流 ====\n")
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，程序退出")
        return None
    
    # 创建所需目录
    create_directories()
    
    # 打印模式信息
    if args.quick:
        logger.info("\n⚠️ 快速测试模式已启用 - 将限制基因数量以加快处理速度")
        logger.info("注意：这将降低结果的准确性，仅用于测试流程\n")
    else:
        logger.info("\n使用完整分析模式 - 将处理所有匹配的基因\n")
    
    # 打印资源信息
    mem = psutil.virtual_memory()
    logger.info(f"系统内存: {mem.total/1e9:.2f} GB 总计, {mem.available/1e9:.2f} GB 可用")
    logger.info(f"内存限制: {args.memory_limit * 100}% 可用内存")
    logger.info(f"CPU核心数: {os.cpu_count()}")
    
    # 定义数据文件路径
    data_files = {
        'pbmc': {
            'h5': 'data/raw/multiome/pbmc/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5',
            'fragments': 'data/raw/multiome/pbmc/pbmc_unsorted_10k_atac_fragments.tsv.gz',
            'peak_annotation': 'data/raw/multiome/pbmc/pbmc_atac_peak_annotations.tsv'
        },
        'brain': {
            'h5': 'data/raw/multiome/brain/filtered_feature_bc_matrix.h5',
            'fragments': 'data/raw/multiome/brain/atac_fragments.tsv.gz',
            'peak_annotation': 'data/raw/multiome/brain/brain_atac_peak_annotations.tsv'
        },
        'jejunum': {
            'h5': 'data/raw/multiome/jejunum/M_Jejunum_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5',
            'fragments': 'data/raw/multiome/jejunum/M_Jejunum_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_atac_fragments.tsv.gz',
            'peak_annotation': 'data/raw/multiome/jejunum/jejunum_atac_peak_annotations.tsv'
        }
    }
    
    # 过滤需要处理的组织
    if args.tissues != 'all':
        tissues_to_process = args.tissues.split(',')
        filtered_data_files = {k: v for k, v in data_files.items() if k in tissues_to_process}
        if not filtered_data_files:
            logger.warning(f"未找到指定的组织: {args.tissues}")
            logger.info(f"可用的组织: {', '.join(data_files.keys())}")
            return None
        data_files = filtered_data_files
    
    # 单细胞数据文件路径
    sc_file = 'data/raw/sc/pbmc/5k_Human_PBMC_TotalSeqB_3p_nextgem_5k_Human_PBMC_TotalSeqB_3p_nextgem_count_sample_filtered_feature_bc_matrix.h5'
    
    # 基因组注释文件
    genome_annotation = 'data/genome/gencode.v47.gtf'
    
    # 检查压缩版本的基因组注释文件
    compressed_annotation = 'data/genome/gencode.v47.chr_patch_hapl_scaff.annotation.gtf.gz'
    
    # 检查基因组注释文件是否存在
    if os.path.exists(genome_annotation):
        logger.info(f"使用基因组注释文件: {genome_annotation}")
    elif os.path.exists(compressed_annotation):
        logger.info(f"使用压缩的基因组注释文件: {compressed_annotation}")
        genome_annotation = compressed_annotation
    else:
        logger.warning(f"基因组注释文件 {genome_annotation} 不存在")
        logger.info("将使用随机基因名注释或峰注释文件")
        genome_annotation = None
    
    # 检查每个组织的峰注释文件
    for tissue, files in data_files.items():
        if 'peak_annotation' in files and files['peak_annotation']:
            if os.path.exists(files['peak_annotation']):
                logger.info(f"找到{tissue}组织的峰注释文件: {files['peak_annotation']}")
            else:
                logger.warning(f"{tissue}组织的峰注释文件 {files['peak_annotation']} 不存在")
                logger.info(f"请下载并放入 {os.path.dirname(files['peak_annotation'])} 目录")
                logger.info("提示: 从数据下载页面获取'ATAC peak annotations based on proximal genes (TSV)'文件")
    
    # 定义标记基因
    marker_genes = {
        'pbmc': {
            'B cell': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'CD20', 'CD22', 'PAX5'],
            'CD14+ monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'CD68', 'FCGR3A'],
            'CD4 T cell': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'IL7R', 'CCR7'],
            'CD8 T cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B'],
            'NK cell': ['GNLY', 'NKG7', 'KLRD1', 'NCAM1', 'CD3D-', 'CD3E-', 'CD3G-'],
            'Dendritic cell': ['CLEC9A', 'CD1C', 'FCER1A', 'LILRA4'],
            'Platelet': ['PPBP', 'PF4']
        },
        'brain': {
            'Excitatory neuron': ['SLC17A7', 'SATB2', 'CUX2', 'RORB', 'THEMIS'],
            'Inhibitory neuron': ['GAD1', 'GAD2', 'DLX1', 'DLX2', 'SST', 'PVALB', 'VIP'],
            'Astrocyte': ['AQP4', 'GFAP', 'ALDH1L1', 'GJA1'],
            'Oligodendrocyte': ['MBP', 'MOG', 'MOBP', 'PLP1'],
            'OPC': ['PDGFRA', 'CSPG4', 'PCDH15'],
            'Microglia': ['CX3CR1', 'P2RY12', 'CSF1R', 'SPI1', 'PTPRC'],
            'Endothelial': ['FLT1', 'CLDN5', 'PECAM1']
        },
        'jejunum': {
            'Enterocyte': ['FABP1', 'ALPI', 'SI', 'LCT', 'ANPEP'],
            'Goblet cell': ['MUC2', 'TFF3', 'SPDEF', 'AGR2'],
            'Paneth cell': ['LYZ', 'DEFA5', 'DEFA6', 'REG3A'],
            'Enteroendocrine cell': ['CHGA', 'CHGB', 'SYP', 'NEUROD1'],
            'Stem cell': ['LGR5', 'OLFM4', 'ASCL2', 'SOX9'],
            'Tuft cell': ['POU2F3', 'TRPM5', 'DCLK1', 'GFI1B'],
            'M cell': ['GP2', 'CCL20', 'SPIB']
        }
    }
    
    # 估计内存需求
    logger.info("估计内存需求...")
    files_to_check = []
    for tissue, files_dict in data_files.items():
        files_to_check.append(files_dict['h5'])
        if 'fragments' in files_dict and files_dict['fragments']:
            files_to_check.append(files_dict['fragments'])
    
    if os.path.exists(sc_file):
        files_to_check.append(sc_file)
    
    estimate_memory_usage(files_to_check, psutil.virtual_memory().total, logger)
    
    # 处理多组学数据
    all_results = {}
    
    # 处理每种组织类型
    progress_total = len(data_files) + (1 if os.path.exists(sc_file) else 0)
    progress_current = 0
    
    for tissue, files in data_files.items():
        progress_current += 1
        logger.info(f"\n进度: [{progress_current}/{progress_total}] - 开始处理 {tissue} 组织数据")
        
        # 检查数据文件是否存在
        if not os.path.exists(files['h5']):
            logger.warning(f"h5文件 {files['h5']} 不存在，跳过 {tissue} 组织数据处理")
            continue
        
        # 分析组织数据
        result = analyze_tissue_data(tissue, files, marker_genes, genome_annotation, args, logger)
        
        if result is not None:
            all_results[tissue] = result
            logger.info(f"成功完成 {tissue} 组织数据分析")
        else:
            logger.error(f"未能完成 {tissue} 组织数据分析")
    
    # 处理单细胞+ADT数据
    if os.path.exists(sc_file):
        progress_current += 1
        logger.info(f"\n进度: [{progress_current}/{progress_total}] - 开始处理单细胞+ADT数据")
        
        # 检查检查点
        checkpoint_name = "pbmc_sc_adt"
        output_dir = "data/processed"
        
        if args.resume:
            sc_data = load_checkpoint(checkpoint_name, output_dir, logger)
            if sc_data is not None:
                logger.info("从检查点恢复单细胞+ADT数据")
                
                cell_counts = {}
                if 'cell_type' in sc_data.obs:
                    cell_counts = sc_data.obs['cell_type'].value_counts().to_dict()
                
                all_results['pbmc_sc'] = {
                    'data': sc_data,
                    'cell_counts': cell_counts
                }
            else:
                try:
                    logger.info("\n=== 开始处理单细胞+ADT数据 ===")
                    sc_data, sc_cell_counts = process_singlecell_adt_data(sc_file)
                    
                    if sc_data is not None:
                        all_results['pbmc_sc'] = {
                            'data': sc_data,
                            'cell_counts': sc_cell_counts
                        }
                        
                        # 保存检查点
                        if args.checkpoint:
                            save_checkpoint(sc_data, checkpoint_name, output_dir, logger)
                        
                        logger.info("单细胞+ADT数据处理完成")
                    else:
                        logger.warning("单细胞+ADT数据处理失败")
                except Exception as e:
                    logger.error(f"处理单细胞+ADT数据时出错: {e}")
                    logger.error(traceback.format_exc())
        else:
            try:
                logger.info("\n=== 开始处理单细胞+ADT数据 ===")
                sc_data, sc_cell_counts = process_singlecell_adt_data(sc_file)
                
                if sc_data is not None:
                    all_results['pbmc_sc'] = {
                        'data': sc_data,
                        'cell_counts': sc_cell_counts
                    }
                    
                    # 保存检查点
                    if args.checkpoint:
                        save_checkpoint(sc_data, checkpoint_name, output_dir, logger)
                    
                    logger.info("单细胞+ADT数据处理完成")
                else:
                    logger.warning("单细胞+ADT数据处理失败")
            except Exception as e:
                logger.error(f"处理单细胞+ADT数据时出错: {e}")
                logger.error(traceback.format_exc())
    else:
        logger.warning(f"单细胞数据文件 {sc_file} 不存在，跳过单细胞+ADT数据处理")
    
    # 汇总和比较结果
    try:
        # 汇总所有数据集的细胞类型信息
        all_cell_counts = []
        
        for dataset, results in all_results.items():
            cell_counts = results.get('cell_counts', {})
            for cell_type, count in cell_counts.items():
                all_cell_counts.append({
                    'dataset': dataset,
                    'cell_type': cell_type, 
                    'count': count
                })
        
        # 创建汇总DataFrame
        if all_cell_counts:
            summary_df = pd.DataFrame(all_cell_counts)
            summary_file = 'data/processed/cell_type_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"\n细胞类型汇总已保存到: {summary_file}")
            
            # 创建可视化摘要
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # 设置可视化样式
                sns.set(style="whitegrid")
                
                # 创建透视表
                pivot_df = summary_df.pivot_table(
                    index='cell_type', 
                    columns='dataset', 
                    values='count', 
                    aggfunc='sum'
                ).fillna(0).astype(int)
                
                # 打印汇总表格
                logger.info("\n=== 细胞类型汇总 ===")
                logger.info(pivot_df)
                
                # 绘制热图
                plt.figure(figsize=(12, 10))
                sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt="d")
                plt.title("细胞类型在不同数据集中的数量")
                plt.tight_layout()
                plt.savefig('data/processed/cell_type_heatmap.png', dpi=300)
                logger.info("细胞类型热图已保存到: data/processed/cell_type_heatmap.png")
                
                # 绘制条形图
                plt.figure(figsize=(14, 8))
                summary_df.pivot_table(
                    index='cell_type', 
                    columns='dataset', 
                    values='count', 
                    aggfunc='sum'
                ).fillna(0).plot(kind='bar')
                plt.title("不同数据集中的细胞类型分布")
                plt.ylabel("细胞数量")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('data/processed/cell_type_barplot.png', dpi=300)
                logger.info("细胞类型条形图已保存到: data/processed/cell_type_barplot.png")
            except Exception as e:
                logger.error(f"创建可视化摘要时出错: {e}")
        else:
            logger.warning("\n没有细胞类型数据可供汇总")
    except Exception as e:
        logger.error(f"创建汇总报告时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 保存分析结果汇总
    try:
        results_summary = {
            'datasets_analyzed': list(all_results.keys()),
            'total_cells': sum(sum(result['cell_counts'].values()) for result in all_results.values()),
            'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'runtime_seconds': time.time() - start_time
        }
        
        summary_json = os.path.join('data/processed', 'analysis_summary.json')
        with open(summary_json, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"分析汇总已保存到: {summary_json}")
    except Exception as e:
        logger.error(f"保存分析汇总时出错: {e}")
    
    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    return all_results

if __name__ == "__main__":
    main() 