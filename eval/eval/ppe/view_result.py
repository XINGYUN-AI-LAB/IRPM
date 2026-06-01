#!/usr/bin/env python3
"""
查看单个评测结果文件的准确率
使用方法: python view_result.py <result_file_path>
"""
import os
os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import argparse
import sys
import json
import os
from pathlib import Path
from multiprocessing import Lock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.scorers import HumanPreferenceScorer
from utils.scoring import init_locks
import pandas as pd
import numpy as np


def format_value(value):
    """格式化数值输出"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_results(results, detailed=False):
    """格式化输出结果"""
    print("=" * 80)
    print("评测结果准确率")
    print("=" * 80)
    
    # 输出总体结果
    overall = results.get("overall", {})
    print("\n【总体结果】")
    print(f"  准确率 (Accuracy):        {format_value(overall.get('accuracy'))}")
    print(f"  Spearman 相关性:          {format_value(overall.get('spearman'))}")
    print(f"  Kendall Tau:              {format_value(overall.get('kendalltau'))}")
    print(f"  Brier Score:              {format_value(overall.get('brier'))}")
    print(f"  Row-wise Pearson:         {format_value(overall.get('row-wise pearson'))}")
    print(f"  Confidence Agreement:      {format_value(overall.get('confidence_agreement'))}")
    print(f"  Separability:             {format_value(overall.get('separability'))}")
    
    # 输出各分类结果（如果启用详细模式）
    if detailed:
        print("\n【分类结果】")
        categories = [
            "hard_prompt",
            "easy_prompt",
            "if_prompt",
            "is_code",
            "math_prompt",
            "shorter_won",
            "similar_response",
            "english_prompt",
            "non_english_prompt",
            "chinese_prompt",
            "russian_prompt",
        ]
        
        for category in categories:
            if category in results:
                cat_result = results[category]
                acc = cat_result.get('accuracy')
                if acc is not None:
                    print(f"  {category:25s}: {format_value(acc)}")
                else:
                    print(f"  {category:25s}: N/A (no samples)")
    
    print("\n" + "=" * 80)


def print_file_stats(file_path):
    """打印文件统计信息"""
    try:
        df = pd.read_json(file_path, lines=True)
        total = len(df)
        has_decision = int(df['decision'].notna().sum()) if 'decision' in df.columns else 0
        has_scores = False
        avg_score_1 = None
        avg_score_2 = None
        
        if 'score_1' in df.columns and 'score_2' in df.columns:
            score_mask = df[['score_1', 'score_2']].notna().all(axis=1)
            has_scores = int(np.sum(score_mask))
            if has_scores > 0:
                avg_score_1 = float(df['score_1'].mean())
                avg_score_2 = float(df['score_2'].mean())
        
        print("\n【文件统计】")
        print(f"  总样本数:                {total}")
        print(f"  有决策的样本:            {has_decision} ({has_decision/total*100:.1f}%)")
        if has_scores:
            print(f"  有分数的样本:            {has_scores} ({has_scores/total*100:.1f}%)")
            print(f"  平均 score_1:            {avg_score_1:.2f}")
            print(f"  平均 score_2:            {avg_score_2:.2f}")
    except Exception as e:
        print(f"\n【文件统计】无法读取: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='查看评测结果文件的准确率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python view_result.py data/human_preference_v1/result.jsonl
  
  # 显示详细分类结果
  python view_result.py data/human_preference_v1/result.jsonl --detailed
  
  # JSON格式输出
  python view_result.py data/human_preference_v1/result.jsonl --json
        """
    )
    parser.add_argument('file_path', type=str, help='结果文件路径 (.jsonl)')
    parser.add_argument('--detailed', action='store_true', help='显示详细分类结果')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出')
    parser.add_argument('--stats', action='store_true', help='显示文件统计信息')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file_path):
        print(f"错误: 文件不存在: {args.file_path}", file=sys.stderr)
        sys.exit(1)
    
    # 检查文件扩展名
    if not args.file_path.endswith('.jsonl'):
        print(f"警告: 文件扩展名不是 .jsonl: {args.file_path}", file=sys.stderr)
    
    # 确保缓存目录存在
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    
    # 初始化锁（用于缓存访问，单进程模式下也需要）
    gt_lock = Lock()
    hp_lock = Lock()
    bok_lock = Lock()
    init_locks(gt_lock, hp_lock, bok_lock)
    
    # 显示文件路径
    print(f"文件路径: {args.file_path}")
    
    # 显示文件统计信息（如果启用）
    if args.stats:
        print_file_stats(args.file_path)
    
    # 计算准确率
    try:
        print("\n正在计算准确率...")
        scorer = HumanPreferenceScorer(args.file_path, is_llm_judge=True)
        results = scorer.score()
        
        # 输出结果
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            format_results(results, args.detailed)
            
    except Exception as e:
        print(f"\n错误: 计算准确率时出错: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

