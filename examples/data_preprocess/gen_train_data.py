#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import sys

import pandas as pd
import numpy as np
import json
import random
# import prompt
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-32B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("./")

# def gen_verl_train_data_grm_follow_filter(input_path, path, input_path2=None):
#     """_summary_
#     Args:
#         type (int, optional): 1: 表示人物2个标准. 0：表示人物统一.
#     """
#     # for subset in ['preference']:
#     subset = 'preference'
#     train_file = os.path.join(input_path, subset, 'train.jsonl')
#     test_file = os.path.join(input_path, subset, 'validation.jsonl')
#     lines = [json.loads(line) for line in open(train_file, 'r')]
#     test_lines = [json.loads(line) for line in open(test_file, 'r')]
#     print(f"train data len: {len(lines)}", f"test data len: {len(test_lines)}")
#     print(type(lines[0]['overall_preference']))
#
#     train_data = {}
#     train_data['problem'] = []
#     train_data['solution'] = []
#
#     test_data = {}
#     test_data['problem'] = []
#     test_data['solution'] = []
#
#     random.seed(42)
#     random.shuffle(lines)
#     idx = 0
#     for d in lines:
#         context = d['context']
#         response1 = d['response1']
#         response2 = d['response2']
#         score = d['overall_preference']
#         if score < -1:
#             chosen = response1
#             rejected = response2
#         elif score > 1:
#             chosen = response2
#             rejected = response1
#         else:
#             print('tie')
#             continue
#         if len(context) % 2 != 1:
#             print(f"context: {len(context)}")
#             continue
#         context_l = []
#         for i, c in enumerate(context):
#             text = "[" + c['role'].upper() + "]" + c['content']
#             context_l.append(text)
#         context_text = '\n'.join(context_l)
#         train_prompt1 = steer_prompt.format(context=context_text, response=chosen)
#         train_prompt2 = steer_prompt.format(context=context_text, response=rejected)
#
#         if len(tokenizer.encode(train_prompt1)) >= 3000 or len(tokenizer.encode(train_prompt2)) >= 3000:
#             print(f'prompt len: {len(train_prompt1)}, {len(train_prompt2)}')
#             continue
#
#         domain = d['domain']
#         language = d['language']
#         solution1 = f"{domain}#-#{language}#-#{idx}_0"
#         solution2 = f"{domain}#-#{language}#-#{idx}_1"
#         train_data['problem'].append(train_prompt1)
#         train_data['solution'].append(solution1)
#         train_data['problem'].append(train_prompt2)
#         train_data['solution'].append(solution2)
#         idx += 1
#     idx = 0
#     for d in test_lines:
#         context = d['context']
#         response1 = d['response1']
#         response2 = d['response2']
#         score = d['overall_preference']
#         if score < 0:
#             chosen = response1
#             rejected = response2
#         elif score > 0:
#             chosen = response2
#             rejected = response1
#         else:
#             print('tie')
#             continue
#         if len(context) % 2 != 1:
#             print(f"context: {len(context)}")
#             continue
#         context_l = []
#         for i, c in enumerate(context):
#             text = "[" + c['role'].upper() + "]" + c['content']
#             context_l.append(text)
#         context_text = '\n'.join(context_l)
#         train_prompt1 = steer_prompt.format(context=context_text, response=chosen)
#         train_prompt2 = steer_prompt.format(context=context_text, response=rejected)
#
#         if len(tokenizer.encode(train_prompt1)) >= 3000 or len(tokenizer.encode(train_prompt2)) >= 3000:
#             print(f'prompt len: {len(train_prompt1)}, {len(train_prompt2)}')
#             continue
#
#         domain = d['domain']
#         language = d['language']
#         solution1 = f"{domain}#-#{language}#-#{idx}_0"
#         solution2 = f"{domain}#-#{language}#-#{idx}_1"
#         test_data['problem'].append(train_prompt1)
#         test_data['solution'].append(solution1)
#         test_data['problem'].append(train_prompt2)
#         test_data['solution'].append(solution2)
#         idx += 1
#     print(len(train_data['problem']), len(test_data['problem']))
#     print('-'*10)
#     print(train_data['problem'][0])
#     print(train_data['solution'][0])
#     print('-'*10)
#     print(train_data['problem'][1])
#     print(train_data['solution'][1])
#     # print(train_data['problem'][-2])
#     # print(train_data['solution'][-10:])
#     dataset = DatasetDict({
#         'train': Dataset.from_dict(train_data),
#         'test': Dataset.from_dict(test_data)
#     })
#     dataset.save_to_disk(path)


def gen_verl_train_data_grm_follow_merge_all(input_dir, path, name_list=[]):
    """_summary_
    Args:
        type (int, optional): 1: 表示人物2个标准. 0：表示人物统一.
    """
    random.seed(42)

    data_list = []
    test_data_list = []

    details = {}

    ## helpsteer3
    for name in name_list:
        if name in ('steer3_filter', ):
            start_len = len(data_list)
            train_file = os.path.join(input_dir, 'preference', 'train.jsonl')
            test_file = os.path.join(input_dir, 'preference', 'validation.jsonl')
            lines = [json.loads(line) for line in open(train_file, 'r')]
            test_lines = [json.loads(line) for line in open(test_file, 'r')]
            print(f"train data len: {len(lines)}", f"test data len: {0}")
            print(type(lines[0]['overall_preference']))

            for idx, d in enumerate(lines):
                context = d['context']
                response1 = d['response1']
                response2 = d['response2']
                score = d['overall_preference']
                if score < -1:
                    chosen = response1
                    rejected = response2
                elif score > 1:
                    chosen = response2
                    rejected = response1
                else:
                    print('tie')
                    continue
                if len(context) % 2 != 1:
                    print(f"context: {len(context)}")
                    continue
                context_l = []
                for i, c in enumerate(context):
                    text = "[" + c['role'].upper() + "]" + c['content']
                    context_l.append(text)
                context_text = '\n'.join(context_l)
                train_prompt1 = steer_prompt.format(context=context_text, response=chosen)
                train_prompt2 = steer_prompt.format(context=context_text, response=rejected)

                if len(tokenizer.encode(train_prompt1)) >= 3000 or len(tokenizer.encode(train_prompt2)) >= 3000:
                    print(f'prompt len: {len(train_prompt1)}, {len(train_prompt2)}')
                    continue

                domain = d['domain']
                language = abs(score)  # d['language']
                solution1 = f"{domain}#-#{language}#-#steer_{idx}_0"
                solution2 = f"{domain}#-#{language}#-#steer_{idx}_1"
                data_list.append([[train_prompt1, train_prompt2], [solution1, solution2]])

            details[name] = len(data_list) - start_len

            for idx, d in enumerate(test_lines):
                context = d['context']
                response1 = d['response1']
                response2 = d['response2']
                score = d['overall_preference']
                if score < -1:
                    chosen = response1
                    rejected = response2
                elif score > 1:
                    chosen = response2
                    rejected = response1
                else:
                    print('tie')
                    continue
                if len(context) % 2 != 1:
                    print(f"context: {len(context)}")
                    continue
                context_l = []
                for i, c in enumerate(context):
                    text = "[" + c['role'].upper() + "]" + c['content']
                    context_l.append(text)
                context_text = '\n'.join(context_l)
                train_prompt1 = steer_prompt.format(context=context_text, response=chosen)
                train_prompt2 = steer_prompt.format(context=context_text, response=rejected)

                if len(tokenizer.encode(train_prompt1)) >= 3000 or len(tokenizer.encode(train_prompt2)) >= 3000:
                    print(f'prompt len: {len(train_prompt1)}, {len(train_prompt2)}')
                    continue

                domain = d['domain']
                language = abs(score)  # d['language']
                solution1 = f"{domain}#-#{language}#-#{idx}_0"
                solution2 = f"{domain}#-#{language}#-#{idx}_1"
                test_data_list.append([[train_prompt1, train_prompt2], [solution1, solution2]])

    train_data = {}
    train_data['problem'] = []
    train_data['solution'] = []

    test_data = {}
    test_data['problem'] = []
    test_data['solution'] = []

    random.shuffle(data_list)
    for t in data_list:
        train_data['problem'].append(t[0][0])
        train_data['problem'].append(t[0][1])
        train_data['solution'].append(t[1][0])
        train_data['solution'].append(t[1][1])

    random.seed(40)
    random.shuffle(data_list)
    for t in data_list:
        train_data['problem'].append(t[0][0])
        train_data['problem'].append(t[0][1])
        train_data['solution'].append(t[1][0])
        train_data['solution'].append(t[1][1])

    # random.seed(44)
    # random.shuffle(data_list)
    # for t in data_list:
    #     train_data['problem'].append(t[0][0])
    #     train_data['problem'].append(t[0][1])
    #     train_data['solution'].append(t[1][0])
    #     train_data['solution'].append(t[1][1])

    if len(test_data_list) > 0:
        for t in test_data_list:
            test_data['problem'].append(t[0][0])
            test_data['problem'].append(t[0][1])
            test_data['solution'].append(t[1][0])
            test_data['solution'].append(t[1][1])
    else:
        for t in data_list[-100:]:
            test_data['problem'].append(t[0][0])
            test_data['problem'].append(t[0][1])
            test_data['solution'].append(t[1][0])
            test_data['solution'].append(t[1][1])

    print(len(train_data['problem']), len(test_data['problem']))
    print('-' * 10)
    print(train_data['problem'][0])
    print(train_data['solution'][0])
    print('-' * 10)
    print(train_data['problem'][1])
    print(train_data['solution'][1])
    print("data len:", details)
    dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data)
    })
    dataset.save_to_disk(path)


if __name__ == '__main__':
    input_dir = sys.argv[1]

    steer_prompt = open("./prompt/steer4.md").read()
    out_path = './data/steer3filter_prompt4_x2_score'
    gen_verl_train_data_grm_follow_merge_all(input_dir, out_path, name_list=['steer3_filter'])