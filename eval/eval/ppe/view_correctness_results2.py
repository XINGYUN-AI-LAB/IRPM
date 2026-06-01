#!/usr/bin/env python3
"""
查看和汇总correctness数据集（五个best_of_k数据集）的评测结果
使用方法: 
  python view_correctness_results.py --model-name <model_name> --output-dir <output_dir>
  python view_correctness_results.py --summary  # 汇总所有数据集的结果
"""
import os
os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import argparse
import sys
import json
import glob
from pathlib import Path
from multiprocessing import Lock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.scorers import ConflictScorer
from utils.scoring import init_locks
import pandas as pd
import numpy as np


# 五个correctness数据集
CORRECTNESS_DATASETS = [
    "mmlu_pro_best_of_k",
    "math_best_of_k",
    "gpqa_best_of_k",
    "ifeval_best_of_k",
    "mbpp_plus_best_of_k",
]


def extract_model_pattern_from_path(model_path):
    """
    从模型路径中提取用于匹配文件名的模式
    
    Args:
        model_path: 模型路径
    
    Returns:
        提取的模式字符串列表，包含多个匹配模式（从精确到宽泛）
    """
    if not model_path:
        return []
    
    model_path_str = str(model_path)
    

    
    # 方法1: 提取最后一个路径组件之后的所有内容（最精确）
    # 例如: global_step_400
    if '/' in model_path_str:
        last_part = model_path_str.split('/')[-1]
        if last_part:
            patterns = [last_part]
    else:
        patterns = [model_path_str]
    
    # 方法2: 提取包含模型标识的关键部分（去除路径前缀）
    path_parts = model_path_str.replace('/', '-').replace('\\', '-').split('-')
    # 过滤掉常见的路径前缀
    filtered_parts = [p for p in path_parts if p and p not in ['home', 'workdir', 'deepspeed', 'work']]
    if filtered_parts:
        # 构建完整模式（包含所有关键部分，特别是 step_xxx）
        full_pattern = '-'.join(filtered_parts)
        if full_pattern not in patterns:
            patterns.append(full_pattern)
        

        key_parts = []
        for i, part in enumerate(filtered_parts):
            if 'step' in part.lower() or (i > 0 and 'step' in filtered_parts[i-1].lower()):
                # 包含 step 的部分及其前后部分
                start_idx = max(0, i-2)
                end_idx = min(len(filtered_parts), i+3)
                key_pattern = '-'.join(filtered_parts[start_idx:end_idx])
                if key_pattern not in patterns:
                    patterns.append(key_pattern)
                break
    
    # 去重并保持顺序
    seen = set()
    unique_patterns = []
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            unique_patterns.append(p)
    
    return unique_patterns if unique_patterns else [model_path_str.replace('/', '-').replace('\\', '-').strip('-').strip()]


def find_result_file(dataset_name, model_name_pattern=None):
    """
    查找指定数据集的结果文件
    
    Args:
        dataset_name: 数据集名称
        model_name_pattern: 模型名称模式（用于匹配文件名）
    
    Returns:
        结果文件路径，如果找不到返回None
    """
    data_dir = Path(__file__).parent / "data" / dataset_name
    
    if not data_dir.exists():
        return None
    
    # 查找pointwise格式的结果文件（排除_raw_outputs文件）
    pattern = f"*pointwise*.jsonl"
    files = list(data_dir.glob(pattern))
    
    # 排除 _raw_outputs.jsonl 和 _debug.log 文件
    files = [f for f in files if '_raw_outputs' not in str(f) and '_debug' not in str(f)]
    
    if not files:
        # 如果没有pointwise文件，查找所有jsonl文件（仍然排除_raw_outputs）
        all_files = list(data_dir.glob("*.jsonl"))
        files = [f for f in all_files if '_raw_outputs' not in str(f) and '_debug' not in str(f)]
    
    if not files:
        return None
    
    # 如果指定了模型名称模式，必须精确匹配
    if model_name_pattern:
        import re
        
        # 首先提取模型路径中的关键信息（特别是 step_xxx）
        model_path_str = str(model_name_pattern)
        
        # 提取 step_xxx 信息（如果存在）
        step_match = re.search(r'step[_-]?(\d+)', model_path_str, re.IGNORECASE)
        target_step = step_match.group(1) if step_match else None
        

        if '/' in model_path_str:
            model_identifier = model_path_str.split('/')[-1]
        else:
            model_identifier = model_path_str
        

        model_identifier_normalized = model_identifier.replace('_', '-') if target_step else model_identifier
        
        matching_files = []
        
        # 精确匹配：文件名必须包含模型标识符的关键部分
        for f in files:
            file_str = str(f)
            
            # 方法1: 如果存在 step_xxx，必须精确匹配 step 数字
            if target_step:
                # 从文件名中提取 step_xxx
                file_step_match = re.search(r'step[_-]?(\d+)', file_str, re.IGNORECASE)
                if file_step_match:
                    file_step = file_step_match.group(1)
                    # step 数字必须完全匹配
                    if file_step != target_step:
                        continue  # 跳过不匹配的文件
            

            # 将模型路径转换为文件名格式（/ -> -）
            model_path_for_filename = model_path_str.replace('/', '-').replace('\\', '-')
            
            # 提取关键部分（去除路径前缀）
            key_parts = []
            parts = model_path_for_filename.split('-')
            skip_prefixes = ['home', 'workdir', 'deepspeed', 'work', '']
            for part in parts:
                if part and part not in skip_prefixes:
                    key_parts.append(part)

            full_pattern = '-'.join(key_parts)
            
            # 检查文件名是否包含完整的模式（或至少大部分关键部分）
            # 优先使用完整模式匹配
            if full_pattern in file_str:
                matching_files.append(f)
            else:
                # 如果完整模式不匹配，检查关键部分匹配度
                # 计算匹配的关键部分数量
                matched_parts = 0
                for part in key_parts:
                    if len(part) > 2 and part in file_str:  # 只匹配长度大于2的部分
                        matched_parts += 1
                
                # 如果匹配的关键部分数量足够（至少70%），则认为匹配
                # 提高阈值以避免误匹配
                if len(key_parts) > 0 and matched_parts >= max(5, len(key_parts) * 0.7):
                    matching_files.append(f)
        
        # 如果找到了匹配的文件
        if matching_files:
            # 去重
            seen = set()
            unique_matching_files = []
            for f in matching_files:
                if str(f) not in seen:
                    seen.add(str(f))
                    unique_matching_files.append(f)
            
            if unique_matching_files:
                # 返回最新的匹配文件
                return str(max(unique_matching_files, key=lambda p: p.stat().st_mtime))
        
        # 如果没有找到匹配的文件，返回 None（而不是返回最新的文件）
        # 这样可以避免返回其他模型的文件
        return None
    
    # 如果没有指定模型名称模式，返回最新的文件
    return str(max(files, key=lambda p: p.stat().st_mtime))


def calculate_accuracy(file_path):
    """计算单个结果文件的准确率"""
    try:
        scorer = ConflictScorer(file_path, k=32, is_llm_judge=True)
        results = scorer.score()
        
        # 返回总体准确率
        if "all" in results:
            return results["all"].get("accuracy")
        else:
            # 如果没有"all"，计算平均值
            accuracies = [r.get("accuracy") for r in results.values() if r.get("accuracy") is not None]
            return np.mean(accuracies) if accuracies else None
    except Exception as e:
        print(f"  错误: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def view_single_dataset(dataset_name, model_name_pattern=None, detailed=False, mode="pairwise"):
    """查看单个数据集的结果"""
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # file_path = find_result_file(dataset_name, model_name_pattern)
    file_path = './math_pointwise.jsonl'

    if not file_path:
        if model_name_pattern:
            print(f"  警告: 未找到匹配模型 '{model_name_pattern}' 的结果文件")
            print(f"  请检查:")
            print(f"    1. 模型路径是否正确: {model_name_pattern}")
            print(f"    2. 结果文件是否存在于: data/{dataset_name}/")
            print(f"    3. 文件名是否包含正确的模型标识符")
        else:
            print(f"  未找到结果文件")
        return None
    
    print(f"  结果文件: {file_path}")
    
    # 显示文件统计信息
    try:
        df = pd.read_json(file_path, lines=True)
        total = len(df)
        has_decision = int(df['decision'].notna().sum()) if 'decision' in df.columns else 0
        has_scores = False
        
        if 'score_1' in df.columns and 'score_2' in df.columns:
            score_mask = df[['score_1', 'score_2']].notna().all(axis=1)
            has_scores = int(np.sum(score_mask))
        
        print(f"  总样本数: {total}")
        print(f"  有决策的样本: {has_decision} ({has_decision/total*100:.1f}%)")
        if has_scores:
            print(f"  有分数的样本: {has_scores} ({has_scores/total*100:.1f}%)")
    except Exception as e:
        print(f"  无法读取文件统计: {e}")
    
    # 计算准确率
    print(f"\n  正在计算准确率...")
    accuracy = calculate_accuracy(file_path)
    
    if accuracy is not None:
        print(f"  准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"  无法计算准确率")
    
    return {
        "dataset": dataset_name,
        "file_path": file_path,
        "accuracy": accuracy,
    }



if __name__ == '__main__':
    view_single_dataset('math_best_of_k', )

