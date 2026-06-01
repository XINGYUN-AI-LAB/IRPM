# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from verl.utils.reward_score.math import compute_score
import re
import numpy as np
from scipy import stats
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import random
import json
import time
from collections import defaultdict, Counter



def apply_func_with_threads(df, prompt, func, column, column2, num_workers=2):
    # model qwen2-72b-instruct , gpt-4o-mini-0718
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(lambda p: func(*p), [row, prompt]) for _, row in df.iterrows()]
        for future in futures:
            results.append(future.result())
    df[column] = [t[0] for t in results]
    df[column2] = [t[1] for t in results]
    return True


def apply_func_with_threads2(df, prompt, func, script_column, score_column, num_workers=2):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(lambda p: func(*p), [row, prompt, script_column]) for _, row in df.iterrows()]
        for future in futures:
            results.append(future.result())

    df[score_column[0]] = [t[0] for t in results]
    df[score_column[1]] = [t[1] for t in results]
    return True


# def parse_point_result(output: str):
#     analysis_pattern = r'\[The Begin of Analysis\]\s*(.*?)\s*\[The End of Analysis\]'
#     analysis_match = re.search(analysis_pattern, output, re.DOTALL)
#     analysis_content = ''
#     if analysis_match:
#         analysis_content = analysis_match.group(1).strip()
#     score_pattern = r'\\boxed\{(\d+(?:\.\d+)?)\}'
#     score_match = re.search(score_pattern, output)
#     try:
#         if score_match:
#             score_str = score_match.group(1)
#             score = float(score_str)
#         else:
#             score = -1.0
#         return analysis_content, score
#     except Exception as e:
#         print(e)
#         return analysis_content, -1.0

def parse_point_result(output: str):
    pattern = r"<critique>\s*(.*?)\s*</critique>\s*<score>\s*(\d+(?:\.\d+)?)\s*</score>\s*"
    match = re.search(pattern, output, re.DOTALL)
    # error_score = 0.0 if idx == 0 else 10.0
    if not match:
        return '', -1.0
    critique_content = match.group(1).strip()
    score_str = match.group(2).strip()
    # # 评语不能为空
    # if not critique_content:
    #     return {'critique': None, 'score': None}
    try:
        score = float(score_str)
        return critique_content, score
    except ValueError:
        return critique_content, -1.0
    # return critique_content, score


def rm_score2(predict_str: str, ground_truth: str, valid_prompt_length, ask_gpt=False):
    input_list = ground_truth.split('#-#')

    domain, language, label_idx = input_list
    idx = int(label_idx.split('_')[-1])
    # length = float(length)
    predict_str = predict_str.replace('<|im_end|>', '')
    format_score = 0.0
    critique, score = parse_point_result(predict_str)
    # print('critique', critique)
    # print('score', score)
    # critique_num = len(critique.split(' '))
    if (valid_prompt_length > 1900 or valid_prompt_length < 50) and len(predict_str) > 1900 and len(predict_str.split(' ')) > 1900:
        format_score = -0.5
        # gpt_score = 0.0 if idx == 0 else 10.0
        # print(f'length: {len(critique)}, gpt_score: {gpt_score}, valid_prompt_length: {valid_prompt_length}, ######{critique}')

    if score == -1.0 or score < 0.0 or score > 10.0:
        format_score = -0.5
        score = 0.0 if idx == 0 else 10.0
    if ask_gpt:
        prompt, answer = call_aimodel(prompt=critic2score_prompt.format(critique=critique), model='gpt-5', temperature=0.5)
        print(f"answer: {answer}")
        try:
            gpt_score = float(answer)
            if gpt_score == -1 or gpt_score < 0.0 or gpt_score > 10.0:
                gpt_score = 0.0 if idx == 0 else 10.0
                format_score = -0.5
                print(f'gpt_score threshold: {gpt_score}')
            else:
                format_score = 0.0
        except Exception as e:
            format_score = -0.5
            gpt_score = 0.0 if idx == 0 else 10.0
            print(f'exception threshold: {gpt_score}')
    else:
        gpt_score = -1.0

    return score, gpt_score, format_score, predict_str, label_idx


def rm_score3(predict_str: str, ground_truth: str, valid_prompt_length, ask_gpt=False):
    input_list = ground_truth.split('#-#')

    domain, language, label_idx = input_list
    idx = int(label_idx.split('_')[-1])
    preference = int(language)
    # length = float(length)
    predict_str = predict_str.replace('<|im_end|>', '')
    format_score = 0.0
    critique, score = parse_point_result(predict_str)
    # print('critique', critique)
    # print('score', score)
    # critique_num = len(critique.split(' '))
    if (valid_prompt_length > 1900 or valid_prompt_length < 50) and len(predict_str) > 1900 and len(predict_str.split(' ')) > 1900:
        format_score = -0.5
        # gpt_score = 0.0 if idx == 0 else 10.0
        # print(f'length: {len(critique)}, gpt_score: {gpt_score}, valid_prompt_length: {valid_prompt_length}, ######{critique}')

    if score == -1.0 or score < 0.0 or score > 10.0:
        format_score = -0.5
        score = 0.0 if idx == 0 else 10.0
    if ask_gpt:
        prompt, answer = call_aimodel(prompt=critic2score_prompt.format(critique=critique), model='gpt-5', temperature=0.5)
        print(f"answer: {answer}")
        try:
            gpt_score = float(answer)
            if gpt_score == -1 or gpt_score < 0.0 or gpt_score > 10.0:
                gpt_score = 0.0 if idx == 0 else 10.0
                format_score = -0.5
                print(f'gpt_score threshold: {gpt_score}')
            else:
                format_score = 0.0
        except Exception as e:
            format_score = -0.5
            gpt_score = 0.0 if idx == 0 else 10.0
            print(f'exception threshold: {gpt_score}')
    else:
        gpt_score = -1.0

    return score, gpt_score, format_score, predict_str, label_idx, preference


def rm_score3_llama(predict_str: str, ground_truth: str, valid_prompt_length, ask_gpt=False):
    input_list = ground_truth.split('#-#')

    domain, language, label_idx = input_list
    idx = int(label_idx.split('_')[-1])
    preference = int(language)
    # length = float(length)
    predict_str = predict_str.replace('<|im_end|>', '')
    format_score = 0.0
    critique, score = parse_point_result(predict_str)
    # print('critique', critique)
    # print('score', score)
    # critique_num = len(critique.split(' '))
    if (valid_prompt_length > 1900 or valid_prompt_length < 50) and len(predict_str) > 1900 and len(predict_str.split(' ')) > 1900:
        format_score = -0.5
        # gpt_score = 0.0 if idx == 0 else 10.0
        # print(f'length: {len(critique)}, gpt_score: {gpt_score}, valid_prompt_length: {valid_prompt_length}, ######{critique}')
    if len(re.findall(r'</critique>', predict_str)) > 1 or len(re.findall(r'</score>', predict_str)) > 1:
        format_score += -0.5

    if score == -1.0 or score < 0.0 or score > 10.0:
        format_score = -0.5
        score = 0.0 if idx == 0 else 10.0
    if ask_gpt:
        prompt, answer = call_aimodel(prompt=critic2score_prompt.format(critique=critique), model='gpt-5', temperature=0.5)
        print(f"answer: {answer}")
        try:
            gpt_score = float(answer)
            if gpt_score == -1 or gpt_score < 0.0 or gpt_score > 10.0:
                gpt_score = 0.0 if idx == 0 else 10.0
                format_score = -0.5
                print(f'gpt_score threshold: {gpt_score}')
            else:
                format_score = 0.0
        except Exception as e:
            format_score = -0.5
            gpt_score = 0.0 if idx == 0 else 10.0
            print(f'exception threshold: {gpt_score}')
    else:
        gpt_score = -1.0

    return score, gpt_score, format_score, predict_str, label_idx, preference


def compute_score_batched_score_mean(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(score_list_a)
            max_b = np.mean(score_list_b)
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_per25(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.25])[0]
            max_b = np.quantile(score_list_b, [0.75])[0]
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_per50(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.5])[0]
            max_b = np.quantile(score_list_b, [0.5])[0]
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_score_meanandper50(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = (np.quantile(score_list_a, [0.5])[0] + np.mean(score_list_a))/2.0
            max_b = (np.quantile(score_list_b, [0.5])[0] + np.mean(score_list_b))/2.0
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean3(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(score_list_a)
            max_b = np.mean(score_list_b)
            # pair reward
            reward_a = [1.0 if score > max_b + 0.2 else 0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a - 0.2 else 0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_score_mean_per75(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.75])[0]
            max_b = np.quantile(score_list_b, [0.25])[0]
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def confidence_interval_t(s, alpha=0.05):
    """
    s: array-like, samples s_i (length G)
    returns: (mu_hat, var_hat_unbiased, (ci_low, ci_high), t_crit, se)
    """
    s = np.asarray(s, dtype=float)
    G = s.size
    if G < 2:
        raise ValueError("Need at least 2 samples to compute unbiased variance and t-interval.")

    mu_hat = s.mean()
    var_hat = s.var(ddof=1)              # unbiased sample variance
    sigma_hat = np.sqrt(var_hat)
    se = sigma_hat / np.sqrt(G)          # standard error

    df = G - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_low = mu_hat - t_crit * se
    ci_high = mu_hat + t_crit * se
    return mu_hat, var_hat, ci_low, ci_high

def compute_score_batched_score_interval(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b)
            # pair reward
            reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_auc(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            # min_a = np.quantile(score_list_a, [0.75])[0]
            # max_b = np.quantile(score_list_b, [0.25])[0]
            # # pair reward
            # reward_a = [1.0 if score > max_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < min_a else -1.0 for score in score_list_b]
            reward_a = [sum([1 if score > score_b else 0 for score_b in score_list_b]) / len(score_list_b) for score in score_list_a]
            reward_b = [sum([1 if score < score_a else 0 for score_a in score_list_a]) / len(score_list_a) for score in score_list_b]

            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    # print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_preference(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            # gpt_score_list_a = id2gpt_score[idx]
            # gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            # min_a = np.quantile(score_list_a, [0.75])[0]
            # max_b = np.quantile(score_list_b, [0.25])[0]
            # # pair reward
            reward_a = [sum([1.0 / (1.0 + np.exp(score_b - score)) for score_b in score_list_b]) / len(score_list_b) for score in score_list_a]
            reward_b = [sum([1.0 / (1.0 + np.exp(score - score_a)) for score_a in score_list_a]) / len(score_list_a) for score in score_list_b]

            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    # print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_gpt_mean(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, True) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()

    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            # score_list_a = id2score[idx]
            # score_list_b = id2score[idx[:-1] + '1']
            gpt_score_list_a = id2gpt_score[idx]
            gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(gpt_score_list_a)
            max_b = np.mean(gpt_score_list_b)
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in gpt_score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in gpt_score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_gpt_per50(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, True) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()

    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            # score_list_a = id2score[idx]
            # score_list_b = id2score[idx[:-1] + '1']
            gpt_score_list_a = id2gpt_score[idx]
            gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.quantile(gpt_score_list_a, [0.5])[0]  # np.mean(gpt_score_list_a)
            max_b = np.quantile(gpt_score_list_b, [0.5])[0]  # np.mean(gpt_score_list_b)
            # pair reward
            reward_a = [1.0 if score > max_b else -1.0 for score in gpt_score_list_a]
            reward_b = [1.0 if score < min_a else -1.0 for score in gpt_score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_rel(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(rm_score2, solution_str, ground_truth, response_length, True) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    counter = Counter()

    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])
        id2gpt_score[idx].append(gpt_scores[i])
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            gpt_score_list_a = id2gpt_score[idx]
            gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(score_list_a)
            max_b = np.mean(score_list_b)
            # pair reward
            reward_a = [1.0 if score > max_b else 0.0 for score in score_list_a]
            reward_b = [1.0 if score < min_a else 0.0 for score in score_list_b]
            counter.update(Counter(reward_a + reward_b))
            # rel
            rel_a = [np.exp(-(score - gpt_score) ** 2 / 2) for (score, gpt_score) in zip(score_list_a, gpt_score_list_a)]
            rel_b = [np.exp(-(score - gpt_score) ** 2 / 2) for (score, gpt_score) in zip(score_list_b, gpt_score_list_b)]
            # merge reward
            reward_merge_a = [r1 * (0.5 + r2) + r3 for r1, r2, r3 in zip(reward_a, rel_a, format_reward_list_a)]
            reward_merge_b = [r1 * (0.5 + r2) + r3 for r1, r2, r3 in zip(reward_b, rel_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")
    print("rewards", rewards)
    print(f"acc: {round(counter[1] / (counter[1] + counter[0]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_adaptive_pre(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(score_list_a)
            max_b = np.mean(score_list_b)
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_mean_adaptive_pre_llama(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3_llama, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(score_list_a)
            max_b = np.mean(score_list_b)
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_gpt_score_mean_adaptive_pre(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, True) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2gpt_score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            # score_list_a = id2score[idx]
            # score_list_b = id2score[idx[:-1] + '1']
            gpt_score_list_a = id2gpt_score[idx]
            gpt_score_list_b = id2gpt_score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            min_a = np.mean(gpt_score_list_a)
            max_b = np.mean(gpt_score_list_b)
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-2)*1.0) else -1.0 for score, pre in zip(gpt_score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-2)*1.0) else -1.0 for score, pre in zip(gpt_score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_score_median_adaptive_pre(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.5])[0]
            max_b = np.quantile(score_list_b, [0.5])[0]
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_median_adaptive2_pre(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.5])[0]
            max_b = np.quantile(score_list_b, [0.5])[0]
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-1)*0.5) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-1)*0.5) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_median_adaptive3_pre(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            min_a = np.quantile(score_list_a, [0.5])[0]
            max_b = np.quantile(score_list_b, [0.5])[0]
            # pair reward
            reward_a = [1.0 if score > (max_b+(pre-1)*1.0) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (min_a-(pre-1)*1.0) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_interval_ada(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.5)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.5)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-1)*0.5) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-1)*0.5) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_interval_ada2(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.5)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.5)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_interval_ada3(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.7)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.7)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_score_interval_ada4(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.8)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.8)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-1)*0.2) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

def compute_score_batched_score_interval_ada5(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.8)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.8)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-2)*1.0) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_interval_ada6(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.8)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.8)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-2)*0.5) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-2)*0.5) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d


def compute_score_batched_score_interval_ada7(data_sources, solution_strs, ground_truths, extra_infos, response_lengths):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    scores, gpt_scores, format_scores, ids, predict_strs, preferences = [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(rm_score3, solution_str, ground_truth, response_length, False) for solution_str, ground_truth, response_length in zip(solution_strs, ground_truths, response_lengths, strict=True)]
        for future in futures:
            score, gpt_score, format_score, predict_str, idx, preference = future.result()
            scores.append(score)
            gpt_scores.append(gpt_score)
            format_scores.append(format_score)
            ids.append(idx)
            predict_strs.append(predict_str)
            preferences.append(preference)

    with open("log.txt", "a", encoding="utf-8") as f:
        for i in range(20):
            f.write("predict_str:{}\nscore:{},gpt_score:{},format_score:{},idx:{}\n########\n".format(predict_strs[i], scores[i], gpt_scores[i], format_scores[i], ids[i]))
    print("ids", ids)
    print("scores", scores)
    # print("gpt_scores", gpt_scores)
    rewards = [-10.0] * len(scores)
    id2score = defaultdict(list)
    id2gpt_score = defaultdict(list)
    id2format_score = defaultdict(list)
    id2batch_idx = defaultdict(list)
    id2pre = defaultdict(list)
    counter = Counter()
    # id2mean = {}
    # id2std = {}
    for i in range(len(scores)):
        idx = ids[i]
        id2score[idx].append(scores[i])  # 容易出bug
        # id2gpt_score[idx].append(gpt_scores[i])  # 容易出bug
        id2format_score[idx].append(format_scores[i])
        id2batch_idx[idx].append(i)
        id2pre[idx].append(preferences[i])

    print("id2score", id2score)
    print("id2format_score", id2format_score)
    print("id2batch_idx", id2batch_idx)

    for idx in id2score.keys():
        if idx.endswith('1'):
            continue
        # if len(id2score[idx]) == 1:
        #     id2mean[idx] = torch.tensor(0.0)
        #     id2std[idx] = torch.tensor(1.0)
        if len(id2score[idx]) >= 1:
            score_list_a = id2score[idx]
            score_list_b = id2score[idx[:-1] + '1']
            pre_list_a = id2pre[idx]
            pre_list_b = id2pre[idx[:-1] + '1']
            format_reward_list_a = id2format_score[idx]
            format_reward_list_b = id2format_score[idx[:-1] + '1']
            idx_list_a = id2batch_idx[idx]
            idx_list_b = id2batch_idx[idx[:-1] + '1']
            # min_a = np.mean(score_list_a)
            # max_b = np.mean(score_list_b)
            _, _, ci_low_a, ci_high_a = confidence_interval_t(score_list_a, alpha=0.9)
            _, _, ci_low_b, ci_high_b = confidence_interval_t(score_list_b, alpha=0.9)
            # pair reward
            # reward_a = [1.0 if score > ci_high_b else -1.0 for score in score_list_a]
            # reward_b = [1.0 if score < ci_low_a else -1.0 for score in score_list_b]

            # min_a = np.quantile(score_list_a, [0.5])[0]
            # max_b = np.quantile(score_list_b, [0.5])[0]
            # # pair reward
            reward_a = [1.0 if score > (ci_high_b+(pre-2)*0.5) else -1.0 for score, pre in zip(score_list_a, pre_list_a)]
            reward_b = [1.0 if score < (ci_low_a-(pre-2)*0.5) else -1.0 for score, pre in zip(score_list_b, pre_list_b)]
            counter.update(Counter(reward_a + reward_b))
            # merge reward
            reward_merge_a = [r1 + r2 for r1, r2 in zip(reward_a, format_reward_list_a)]
            reward_merge_b = [r1 + r2 for r1, r2 in zip(reward_b, format_reward_list_b)]

            for t_idx, reward_merge in zip(idx_list_a, reward_merge_a):
                rewards[t_idx] = reward_merge
            for t_idx, reward_merge in zip(idx_list_b, reward_merge_b):
                rewards[t_idx] = reward_merge

        else:
            raise ValueError(f"no score in prompt index: {idx}")

    print(f"acc: {round(counter[1] / (counter[1] + counter[-1]), 3)}")
    if any([r == -10.0 for r in rewards]):
        raise ValueError(f"reward error")

    reward_d = [{"score": score, "p_score": p_score, "gpt_score": gpt_score, "format_score": format_score} for score, p_score, gpt_score, format_score in zip(rewards, scores, gpt_scores, format_scores)]

    return reward_d

