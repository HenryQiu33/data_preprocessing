#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块 - 提供依赖检查、文件索引和安全绘图功能
"""

import os
import subprocess
import warnings
import matplotlib.pyplot as plt
import scanpy as sc
import time
import psutil
import numpy as np
import logging
import sys
from pathlib import Path
import hashlib
warnings.filterwarnings('ignore')

def check_dependencies():
    """检查并打印依赖库版本信息"""
    try:
        import pkg_resources
        
        # 关键依赖
        required_packages = {
            'scanpy': '1.8.0',
            'anndata': '0.8.0',
            'pyranges': '0.0.117',
            'pysam': '0.19.0',
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'matplotlib': '3.4.0',
            'scipy': '1.7.0',
            'protobuf': '3.20.0',
            'psutil': '5.9.0'
        }
        
        missing_packages = []
        outdated_packages = []
        
        print("依赖库版本信息:")
        for package, min_version in required_packages.items():
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"  {package}: {version}")
                
                # 检查版本是否满足最低要求
                if version.split('.')[0] < min_version.split('.')[0]:
                    outdated_packages.append(f"{package} (当前: {version}, 需要: {min_version}+)")
                
            except pkg_resources.DistributionNotFound:
                print(f"  {package}: 未安装")
                missing_packages.append(package)
        
        # 特别检查protobuf版本
        try:
            import google.protobuf
            protobuf_version = pkg_resources.get_distribution('protobuf').version
            if protobuf_version.startswith('4.'):
                print("警告: protobuf 4.x可能与某些库不兼容，如果遇到'MessageFactory'错误，请尝试降级到3.20.x")
                print("  pip install protobuf==3.20.3")
        except:
            pass
        
        # 报告缺失或过时的依赖
        if missing_packages:
            print(f"\n警告: 缺少以下依赖: {', '.join(missing_packages)}")
            print("请使用pip安装:")
            print(f"  pip install {' '.join(missing_packages)}")
            return False
            
        if outdated_packages:
            print(f"\n警告: 以下依赖版本过旧: {', '.join(outdated_packages)}")
            print("请考虑更新这些包:")
            print(f"  pip install --upgrade {' '.join([p.split(' ')[0] for p in outdated_packages])}")
        
        return True
            
    except:
        print("无法检查依赖库版本")
        return False

def check_file_integrity(file_path):
    """检查文件完整性"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    # 检查文件大小
    size = os.path.getsize(file_path)
    if size == 0:
        print(f"警告: 文件大小为0: {file_path}")
        return False
    
    # 对于h5文件，尝试验证是否可以读取
    if file_path.endswith('.h5'):
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                if not list(f.keys()):
                    print(f"警告: h5文件无有效数据: {file_path}")
                    return False
        except Exception as e:
            print(f"警告: 无法读取h5文件 {file_path}: {e}")
            return False
    
    # 对于gzip文件，尝试读取前几行
    if file_path.endswith('.gz'):
        try:
            import gzip
            with gzip.open(file_path, 'rt') as f:
                for i in range(5):
                    if not next(f, None):
                        print(f"警告: gzip文件可能已损坏: {file_path}")
                        return False
        except Exception as e:
            print(f"警告: 无法读取gzip文件 {file_path}: {e}")
            return False
    
    return True

def calculate_file_hash(file_path, chunk_size=8192):
    """计算文件的MD5哈希值"""
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"无法计算文件哈希值 {file_path}: {e}")
        return None

def check_and_create_tabix_index(fragments_file):
    """检查并创建fragments文件的tabix索引"""
    index_file = fragments_file + '.tbi'
    
    # 如果索引不存在或比数据文件旧
    if not os.path.exists(index_file) or os.path.getmtime(index_file) < os.path.getmtime(fragments_file):
        print(f"为{fragments_file}创建tabix索引...")
        try:
            # 确认文件存在且有效
            if not check_file_integrity(fragments_file):
                print("fragments文件完整性检查失败，跳过索引创建")
                return False
                
            # 创建索引
            subprocess.run(['tabix', '-p', 'bed', fragments_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("索引创建成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"tabix索引创建失败: {e.stderr.decode() if e.stderr else ''}")
            print("请确保tabix已安装并可用，且fragments文件格式正确")
            return False
        except FileNotFoundError:
            print("找不到tabix命令，请安装tabix")
            print("Ubuntu/Debian: sudo apt-get install tabix")
            print("macOS (Homebrew): brew install tabix")
            print("Conda: conda install -c bioconda tabix")
            return False
    return True

def safe_plot_umap(adata, color, filename_suffix, figsize=(12, 10), dpi=300):
    """安全的UMAP绘图函数，捕获并处理可能的错误"""
    try:
        sc.pl.umap(adata, color=color, save=filename_suffix, figsize=figsize, dpi=dpi)
    except AttributeError as e:
        if "'ColormapRegistry' object has no attribute 'get_cmap'" in str(e):
            print("遇到matplotlib颜色映射错误，尝试替代方法...")
            try:
                # 使用基本的matplotlib代替
                fig, ax = plt.subplots(figsize=figsize)
                
                # 基本散点图
                if 'X_umap' in adata.obsm:
                    x, y = adata.obsm['X_umap'].T
                    
                    if isinstance(color, list):
                        color_value = color[0]  # 使用第一个颜色
                    else:
                        color_value = color
                        
                    if color_value in adata.obs:
                        # 分类数据
                        categories = adata.obs[color_value].cat.categories if hasattr(adata.obs[color_value], 'cat') else adata.obs[color_value].unique()
                        for cat in categories:
                            mask = adata.obs[color_value] == cat
                            ax.scatter(x[mask], y[mask], s=5, alpha=0.7, label=cat)
                        ax.legend()
                    else:
                        # 默认绘图
                        ax.scatter(x, y, s=5, alpha=0.7)
                    
                    ax.set_title(f'UMAP - {color_value}')
                    plt.tight_layout()
                    plt.savefig(f'figures/umap_{filename_suffix.replace("_", "")}.png', dpi=dpi)
                    plt.close()
            except Exception as e2:
                print(f"备用绘图方法也失败: {e2}")
        else:
            print(f"绘图出错: {e}")
    except Exception as e:
        print(f"绘图出错: {e}")

def memory_monitor(threshold=0.8, interval=5.0, callback=None):
    """
    内存监控函数 - 在子线程中运行，定期检查内存使用情况
    
    参数:
    - threshold: 内存使用警告阈值 (0.0-1.0)
    - interval: 检查间隔(秒)
    - callback: 达到阈值时调用的回调函数
    """
    import threading
    
    def monitor_loop():
        while True:
            try:
                mem = psutil.virtual_memory()
                usage = mem.percent / 100.0
                
                if usage > threshold:
                    print(f"\n警告: 内存使用率达到 {usage:.2%}, 阈值为 {threshold:.2%}")
                    print(f"可用内存: {mem.available / 1e9:.2f} GB, 总内存: {mem.total / 1e9:.2f} GB")
                    
                    if callback:
                        callback(usage)
                
                time.sleep(interval)
            except:
                # 忽略错误，继续监控
                time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread

def create_directories():
    """创建项目所需的目录结构"""
    dirs = [
        'data/processed',
        'data/processed/checkpoints',
        'figures',
        'logs',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录结构已创建")

def set_logger(log_file=None, level=logging.INFO):
    """配置日志记录器"""
    logger = logging.getLogger('multiome_analysis')
    logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出(如果提供)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def estimate_resources(file_paths, feature_counts=None, cell_counts=None):
    """
    估计数据处理所需的资源
    
    参数:
    - file_paths: 数据文件路径列表
    - feature_counts: 特征数量(如果已知)
    - cell_counts: 细胞数量(如果已知)
    
    返回:
    - 估计内存使用量(GB)
    - 估计处理时间(分钟)
    """
    # 初始估计
    total_file_size = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
    memory_estimate = total_file_size * 5 / 1e9  # 转换为GB
    
    # 如果知道细胞和特征数量，提供更精确的估计
    if cell_counts and feature_counts:
        # 每个细胞特征矩阵的内存成本(float32)
        matrix_size = cell_counts * feature_counts * 4 / 1e9  # 4字节/float32，转换为GB
        # 额外成本，包括降维、整合等
        extra_size = cell_counts * 100 * 4 / 1e9  # 每个细胞~100个额外值
        memory_estimate = max(memory_estimate, matrix_size + extra_size)
    
    # 粗略估计处理时间
    # 假设处理1GB数据需要约5分钟
    time_estimate = (total_file_size / 1e9) * 5
    
    # 如果知道细胞数量，调整时间估计
    if cell_counts:
        # 假设处理10,000个细胞需要10分钟
        cell_time = (cell_counts / 10000) * 10
        time_estimate = max(time_estimate, cell_time)
    
    return memory_estimate, time_estimate

def print_resource_usage(stage_name="当前"):
    """打印当前资源使用情况"""
    try:
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        process = psutil.Process()
        process_mem = process.memory_info().rss / 1e9  # GB
        
        print(f"\n--- {stage_name}资源使用 ---")
        print(f"进程内存: {process_mem:.2f} GB")
        print(f"系统内存: 使用率 {mem.percent}%, 可用 {mem.available/1e9:.2f} GB, 总计 {mem.total/1e9:.2f} GB")
        print(f"CPU使用率: {cpu_percent}%")
    except:
        print("无法获取资源使用信息")

def generate_figures_summary(figures_dir='figures', output_file='figures/summary.html'):
    """生成图表摘要HTML文件"""
    try:
        figures_path = Path(figures_dir)
        if not figures_path.exists():
            print(f"图表目录 {figures_dir} 不存在")
            return
        
        # 获取所有图片文件
        image_files = sorted([f for f in figures_path.glob('*.png') if f.is_file()])
        image_files.extend(sorted([f for f in figures_path.glob('*.pdf') if f.is_file()]))
        
        if not image_files:
            print("未找到任何图片文件")
            return
        
        # 创建HTML内容
        html = ['<!DOCTYPE html>', '<html><head><title>分析图表摘要</title>',
                '<style>',
                'body { font-family: Arial, sans-serif; margin: 20px; }',
                '.figure { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }',
                '.figure img { max-width: 100%; height: auto; }',
                '.figure h3 { margin-top: 0; }',
                '</style>',
                '</head><body>',
                '<h1>分析图表摘要</h1>',
                f'<p>生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>']
        
        # 添加每个图片
        for img_file in image_files:
            file_name = img_file.name
            rel_path = img_file.relative_to(figures_path)
            title = file_name.split('.')[0].replace('_', ' ').title()
            
            html.append(f'<div class="figure">')
            html.append(f'<h3>{title}</h3>')
            html.append(f'<p>文件: {file_name}</p>')
            
            if file_name.endswith('.pdf'):
                html.append(f'<p><a href="{rel_path}" target="_blank">查看PDF</a></p>')
            else:
                html.append(f'<img src="{rel_path}" alt="{title}">')
            
            html.append('</div>')
        
        html.append('</body></html>')
        
        # 写入HTML文件
        with open(output_file, 'w') as f:
            f.write('\n'.join(html))
        
        print(f"图表摘要已保存到: {output_file}")
    except Exception as e:
        print(f"生成图表摘要时出错: {e}") 