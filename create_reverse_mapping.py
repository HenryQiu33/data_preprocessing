#!/usr/bin/env python3
"""
创建反向基因映射文件，将ENSEMBL ID到基因符号的映射转换为基因符号到ENSEMBL ID的映射
"""

import os
import argparse

def create_reverse_mapping(input_file, output_file):
    """
    创建反向映射文件
    
    Parameters:
    -----------
    input_file : str
        输入映射文件路径（ENSEMBL ID到基因符号）
    output_file : str
        输出映射文件路径（基因符号到ENSEMBL ID）
    """
    reverse_mapping = {}
    duplicate_count = 0
    
    # 读取输入文件
    with open(input_file, 'r') as f:
        # 读取标题行
        header = f.readline().strip().split('\t')
        if len(header) < 2:
            print(f"错误: 输入文件格式不正确，标题行: {header}")
            return
        
        ensembl_idx = header.index('ensembl_id') if 'ensembl_id' in header else 0
        symbol_idx = header.index('gene_symbol') if 'gene_symbol' in header else 1
        
        print(f"映射列: ensembl_id={ensembl_idx}, gene_symbol={symbol_idx}")
        
        # 读取并处理每一行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= max(ensembl_idx, symbol_idx) + 1:
                ensembl_id = parts[ensembl_idx]
                gene_symbol = parts[symbol_idx]
                
                # 跳过没有基因符号的行
                if not gene_symbol or gene_symbol == 'NA' or gene_symbol == '.':
                    continue
                
                # 处理重复的基因符号
                if gene_symbol in reverse_mapping:
                    duplicate_count += 1
                    # 可以选择保留现有映射，或者更新为新的映射
                    # 这里我们选择保留现有映射
                else:
                    reverse_mapping[gene_symbol] = ensembl_id
    
    print(f"从{input_file}读取了{len(reverse_mapping)}个唯一基因符号，发现{duplicate_count}个重复")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.write("gene_symbol\tensembl_id\n")
        for gene_symbol, ensembl_id in sorted(reverse_mapping.items()):
            f.write(f"{gene_symbol}\t{ensembl_id}\n")
    
    print(f"反向映射已保存到{output_file}")

def main():
    parser = argparse.ArgumentParser(description='创建反向基因映射')
    parser.add_argument('--input', type=str, default='data/standard_gene_mapping.tsv',
                        help='输入映射文件路径（ENSEMBL ID到基因符号）')
    parser.add_argument('--output', type=str, default='data/reverse_gene_mapping.tsv',
                        help='输出映射文件路径（基因符号到ENSEMBL ID）')
    
    args = parser.parse_args()
    create_reverse_mapping(args.input, args.output)

if __name__ == "__main__":
    main() 