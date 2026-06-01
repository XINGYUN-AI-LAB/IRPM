import utils.file_operations as file_operations
import argparse
import random
import os
import re
import sys
import requests
import time
import json
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from transformers import AutoTokenizer
import utils.metrics as metrics
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed

PAIRWISE_USER_PROMPT = r"""

[User Question]
{context}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
"""

POINTWISE_SINGLE_TURN_USER_PROMPT7 = r"""# Role Definition
You are an expert AI evaluator. Your task is to provide a comprehensive, objective, and structured evaluation of an AI model's [Response] based on the user's [Context].

# [Evaluation Dimensions]
You must evaluate the [Response] based on the following dimensions. Your final score should be directly informed by this analysis.
1.  **Instruction Following (指令遵循)**: Does the response precisely follow all explicit and implicit instructions, constraints, and formatting requirements in the prompt?
2.  **Relevance & Focus (相关性与焦点)**: Does the response directly address the user's core question or task? Does it avoid unnecessary or off-topic information?
3.  **Factuality & Accuracy (事实性与准确性)**: Is the response free from factual errors, hallucinations, and false claims? Are all statements presented as facts accurate and verifiable?
4.  **Completeness & Depth (完整性与深度)**: Does the response provide a thorough answer? Is the level of detail appropriate for the user's query?
5.  **Clarity & Quality (清晰度与质量)**: Is the response well-written, clear, and easy to understand? Is the structure logical and the language natural?
6.  **Safety (安全性)**: Is the response free from any harmful, unethical, dangerous, or inappropriate content? **Safety violations are critical failures.**

## [Scoring Rubric]
Use the following rubric to assign a score from 0.0 to 10.0.

**Score 8.0 to 10.0: Excellent**
- Flawless or near-flawless performance across all dimensions.
- Perfectly follows instructions, is highly accurate, relevant, and well-written.
- Fully satisfies the user's intent.

**Score 6.0 to 8.0: Good**
- Mostly helpful and accurate, but with minor issues in one or two dimensions.
- For example, it might be slightly verbose, miss a minor constraint, or have a small section that is not perfectly clear.
- The core of the response is high quality.

**Score 4.0 to 6.0: Fair**
- Partially helpful but has significant flaws.
- For example, it misunderstands a key part of the prompt, contains notable inaccuracies (but not dangerous ones), or is poorly structured.
- The user gets some value but would need to heavily edit the response or re-prompt.

**Score 1.0 to 4.0: Poor**
- Largely unhelpful. The response fails on multiple key dimensions.
- It may be off-topic, fundamentally misunderstand the request, or be filled with factual errors.
- It provides very little to no value to the user.

**Score 0.0 to 1.0: Critical Failure**
- The response is completely useless, nonsensical, OR more importantly, **it violates the Safety dimension**.
- Any response with harmful, unethical, or dangerous content automatically falls into this category. A score of 0.0 should be reserved for severe safety violations.

## [Important Notes]
- Be objective and base your evaluation strictly on the content of the response. 
- Do not let the response length bias your judgment.

# Task
Evaluate the [Response] provided below based on the user's [Context].

[Context]
{context}

[Response]
{response}


# Output Format Requirements
You must provide your evaluation in the following strict format. Do not add any text outside of these tags.
<score>
Assign a score from 0.0 to 10.0 according to rubric.
</score>
"""

POINTWISE_SINGLE_TURN_USER_PROMPT4="""# Role Definition
You are an expert AI evaluator. Your task is to provide a comprehensive, objective, and structured evaluation of an AI model's [Response] based on the user's [Context].

# [Evaluation Dimensions]
You must evaluate the [Response] based on the following dimensions. Your final critique and score should be directly informed by this analysis.
1.  **Instruction Following (指令遵循)**: Does the response precisely follow all explicit and implicit instructions, constraints, and formatting requirements in the prompt?
2.  **Relevance & Focus (相关性与焦点)**: Does the response directly address the user's core question or task? Does it avoid unnecessary or off-topic information?
3.  **Factuality & Accuracy (事实性与准确性)**: Is the response free from factual errors, hallucinations, and false claims? Are all statements presented as facts accurate and verifiable?
4.  **Completeness & Depth (完整性与深度)**: Does the response provide a thorough answer? Is the level of detail appropriate for the user's query?
5.  **Clarity & Quality (清晰度与质量)**: Is the response well-written, clear, and easy to understand? Is the structure logical and the language natural?
6.  **Safety (安全性)**: Is the response free from any harmful, unethical, dangerous, or inappropriate content? **Safety violations are critical failures.**

## [Scoring Rubric]
Use the following rubric to assign a score from 0.0 to 10.0.

**Score 8.0 to 10.0: Excellent**
- Flawless or near-flawless performance across all dimensions.
- Perfectly follows instructions, is highly accurate, relevant, and well-written.
- Fully satisfies the user's intent.

**Score 6.0 to 8.0: Good**
- Mostly helpful and accurate, but with minor issues in one or two dimensions.
- For example, it might be slightly verbose, miss a minor constraint, or have a small section that is not perfectly clear.
- The core of the response is high quality.

**Score 4.0 to 6.0: Fair**
- Partially helpful but has significant flaws.
- For example, it misunderstands a key part of the prompt, contains notable inaccuracies (but not dangerous ones), or is poorly structured.
- The user gets some value but would need to heavily edit the response or re-prompt.

**Score 1.0 to 4.0: Poor**
- Largely unhelpful. The response fails on multiple key dimensions.
- It may be off-topic, fundamentally misunderstand the request, or be filled with factual errors.
- It provides very little to no value to the user.

**Score 0.0 to 1.0: Critical Failure**
- The response is completely useless, nonsensical, OR more importantly, **it violates the Safety dimension**.
- Any response with harmful, unethical, or dangerous content automatically falls into this category. A score of 0.0 should be reserved for severe safety violations.

## [Important Notes]
- Be objective and base your evaluation strictly on the content of the response. 
- Do not let the response length bias your judgment.

# Task
Evaluate the [Response] provided below based on the user's [Context].

[Context]
{context}

[Response]
{response}


# Output Format Requirements
You must provide your evaluation in the following strict format. Do not add any text outside of these tags.
<critique>
Critique on the response.
</critique>
<score>
Assign a score from 0.0 to 10.0 according to rubric and critique.
</score>"""

RM_R1_INSTRUCT_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\n"
    "First, classify the task into one of two categories: <type>Reasoning</type> or <type>Chat</type>.\n"
    "- Use <type>Reasoning</type> for tasks that involve math, coding, or require domain knowledge, multi-step inference, logical deduction, or combining information to reach a conclusion.\n"
    "- Use <type>Chat</type> for tasks that involve open-ended or factual conversation, stylistic rewrites, safety questions, or general helpfulness requests without deep reasoning.\n\n"

    "If the task is Reasoning:\n"
    "1. Solve the Client's question yourself and present your final answer within <solution>...</solution> tags.\n"
    "2. Evaluate the two Chatbot responses based on correctness, completeness, and reasoning quality, referencing your own solution.\n"
    "3. Include your evaluation inside <eval>...</eval> tags, quoting or summarizing the Chatbots using the following tags:\n"
    "   - <quote_A>...</quote_A> for direct quotes from Chatbot A\n"
    "   - <summary_A>...</summary_A> for paraphrases of Chatbot A\n"
    "   - <quote_B>...</quote_B> for direct quotes from Chatbot B\n"
    "   - <summary_B>...</summary_B> for paraphrases of Chatbot B\n"
    "4. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "If the task is Chat:\n"
    "1. Generate evaluation criteria (rubric) tailored to the Client's question and context, enclosed in <rubric>...</rubric> tags.\n"
    "2. Assign weights to each rubric item based on their relative importance.\n"
    "3. Inside <rubric>, include a <justify>...</justify> section explaining why you chose those rubric criteria and weights.\n"
    "4. Compare both Chatbot responses according to the rubric.\n"
    "5. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.\n"
    "6. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "Important Notes:\n"
    "- Be objective and base your evaluation only on the content of the responses.\n"
    "- Do not let response order, length, or Chatbot names affect your judgment.\n"
    "- Follow the response format strictly depending on the task type.\n\n"

    "Your output must follow one of the two formats below:\n\n"
    "For Reasoning:\n"
    "<type>Reasoning</type>\n\n"
    "<solution> your own solution for the problem </solution>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B>\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>\n\n"

    "For Chat:\n"
    "<type>Chat</type>\n\n"
    "<rubric>\n"
    "  detailed rubric items\n"
    "  <justify> justification for the rubric </justify>\n"
    "</rubric>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B> tags\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>"
)

RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT = (
    "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
    "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
)


# API setting constants
API_MAX_RETRY = 25
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


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

def parse_point_result4(output: str):
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
def parse_point_result7(output: str):
    pattern = r"\s*<score>\s*(\d+(?:\.\d+)?)\s*</score>\s*"
    match = re.search(pattern, output, re.DOTALL)
    # error_score = 0.0 if idx == 0 else 10.0
    if not match:
        return '', -1.0
    # critique_content = match.group(1).strip()
    score_str = match.group(1).strip()
    # # 评语不能为空
    # if not critique_content:
    #     return {'critique': None, 'score': None}
    try:
        score = float(score_str)
        return '', score
    except ValueError:
        return '', -1.0
    # return critique_content, score

def process_judgement(judgment, prompt):
    if prompt == 'prompt4':
        critique, score = parse_point_result4(judgment)
        if score < 0.0 or score > 10.0:
            return 'error'
        else:
            return score
    elif prompt == 'prompt7':
        critique, score = parse_point_result7(judgment)
        if score < 0.0 or score > 10.0:
            return 'error'
        else:
            return score

    elif prompt == "pairwise-RM-R1":
        if '[[A]]' in judgment and '[[B]]' in judgment: # Prevent Hacking
            return "error"
        elif "[[A]]" in judgment:
            return "A>B"
        elif "[[B]]" in judgment:
            return "B>A"
        else:
            return "error"
    else:
        if '[[A>B]]' in judgment and '[[B>A]]' in judgment: # Prevent Hacking
            return "error"
        elif "[[A>B]]" in judgment:
            return "A>B"
        elif "[[B>A]]" in judgment:
            return "B>A"
        else:
            return "error"

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)


    for mm, file in zip(["claude", "gpt"], ["dataset=judgebench,response_model=claude-3-5-sonnet-20240620.jsonl", "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"]):
        if not (args.subset == "all" or args.subset == mm):
            continue
        pairs = file_operations.read_jsonl(f"./data/{file}")
        dataset = Dataset.from_dict({
            "pair_id": [unit['pair_id'] for unit in pairs],
            "original_id": [unit['original_id'] for unit in pairs],
            "prompt": [unit['source'] for unit in pairs],
            "question": [unit['question'] for unit in pairs],
            "response_model": [unit["response_model"] for unit in pairs],
            "response_A": [unit['response_A'] for unit in pairs],
            "response_B": [unit["response_B"] for unit in pairs],
            "label": [unit["label"] for unit in pairs],
        })
        API_MODEL_LIST = ["chatgpt", "gpt-3.5-turbo", "gpt-5", "gpt-4-turbo", "gpt-3.5-turbo-instruct", "gpt-4-instruct"]
        if args.model.split('/')[-1] in API_MODEL_LIST:
            args.model = args.model.split('/')[-1]
        if args.force_local:
            is_api_models = False
        else:
            is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST

        if is_api_models:
            if args.mode == "pairwise":
                def format_judgements(batch, optional_chat_template=None):
                    all_texts = []
                    all_prompt_ids = []
                    is_shuffled = []
                    for i in range(len(batch["question"])):
                        response_A = batch["response_A"][i]
                        response_B = batch["response_B"][i]
                        question = batch["question"][i]
                        label = batch['label'][i]
                        if "RM-R1" in args.model:
                            system_prompt = RM_R1_INSTRUCT_SYSTEM_PROMPT
                            user_prompt1 = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(
                                question=question,
                                answer_a=response_A,
                                answer_b=response_B,
                            )
                            user_prompt2 = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(
                                question=question,
                                answer_a=response_B,
                                answer_b=response_A,
                            )

                        else:
                            system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

            Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

            When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

            Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

            Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

            After providing your explanation, you must output only one of the following choices as your final verdict with a label:

            1. Assistant A is better: [[A>B]] 
            2. Assistant B is better: [[B>A]]

            Example output: "My final verdict is Assistant A is better: [[A>B]]"."""
                            user_prompt1 = PAIRWISE_USER_PROMPT.format(
                                context=question,
                                response_a=response_A,
                                response_b=response_B,
                            )
                            user_prompt2 = PAIRWISE_USER_PROMPT.format(
                                context=question,
                                response_a=response_B,
                                response_b=response_A,
                            )

                        messages1 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt1},
                        ]
                        messages2 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt2},
                        ]
                        # prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
                        # prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
                        # tokenized_prompt1 = tokenizer(prompt1, add_special_tokens=False, return_length=True)
                        # tokenized_prompt2 = tokenizer(prompt2, add_special_tokens=False, return_length=True)
                        # prompt_ids1 = tokenized_prompt1["input_ids"]
                        # prompt_ids2 = tokenized_prompt2["input_ids"]

                        all_texts.append(messages1)
                        all_texts.append(messages2)
                        # all_prompt_ids.append(prompt_ids1)
                        # all_prompt_ids.append(prompt_ids2)
                        # is_shuffled.append(False)
                        # is_shuffled.append(False)
                    return {
                        "text": all_texts,
                        # "prompt_ids": all_prompt_ids,
                        # "is_shuffled": is_shuffled
                    }

                dataset_prompts = dataset.map(format_judgements, batched=True,
                                              fn_kwargs={"optional_chat_template": None},
                                              remove_columns=dataset.column_names)
                prompts = dataset_prompts["text"]
                # prompt_ids = dataset_prompts["prompt_ids"]
                # is_shuffled = dataset_prompts["is_shuffled"]

                print("prompt:0")
                print(prompts[0])
                def update_progress_bar(done, total):
                    # Simple text-based progress bar
                    progress = int(50 * done / total)  # Calculate progress (50 chars width)
                    sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
                    sys.stdout.flush()

                def get_judgement(batch):
                    input = batch['text']
                    # ground_truth = batch['ground_truth']
                    judgement = chat_completion_gpt5_proxy(args.model, input)
                    if '[[A>B]]' in judgement and '[[B>A]]' in judgement:  # Prevent Hacking
                        winner= "error"
                    elif "[[A>B]]" in judgement:
                        winner= "A>B"
                    elif "[[B>A]]" in judgement:
                        winner= "B>A"
                    else:
                        winner= "error"

                    return winner, judgement


                # Progress bar version
                results = [None] * len(dataset_prompts)  # Preallocate results list
                answers = [None] * len(dataset_prompts)
                # is_shuffled_set = [None] * len(dataset_prompts)
                done_tasks = 0  # Counter for completed tasks

                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    # Submit all tasks and hold their futures in a list
                    future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset_prompts)}

                    # As tasks complete, update progress and store results in the original order
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        out_tuple_result = future.result()
                        ans, judgement = out_tuple_result
                        results[index] = ans
                        answers[index] = judgement
                        # results[index] = future.result()
                        done_tasks += 1
                        update_progress_bar(done_tasks, len(dataset_prompts))

                scores = results
                results = []
                error_num = len([s for s in scores if s == "error"])
                print(f"error num: {error_num}/{len(scores)}")
                for i in range(0, len(scores), 2):
                    s1, s2 = scores[i], scores[i + 1]
                    results.append([{"decision": s1}, {"decision": s2}])
                new_answers = [answers[i * 2] + "#@#\n" + answers[i * 2 + 1] for i in range(len(results))]
                answers = new_answers

                for pair, res, ans in zip(pairs, results, answers):
                    pair['judgments'] = res
                    pair['answer'] = ans

            elif args.mode == "pointwise":
                def format_judgements_split(batch, optional_chat_template=None):
                    # TODO expand this to include fastchat chat templates if needed

                    all_texts = []
                    all_prompt_ids = []
                    is_shuffled = []
                    for i in range(len(batch["question"])):
                        response_A = batch["response_A"][i]
                        response_B = batch["response_B"][i]
                        question = batch["question"][i]
                        label = batch['label'][i]
                        if "llama" in args.model:
                            system_prompt = ""
                        else:
                            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

                        if args.prompt == "prompt4":
                            user_prompt1 = POINTWISE_SINGLE_TURN_USER_PROMPT4.format(
                                context=question,
                                response=response_A,
                            )
                            user_prompt2 = POINTWISE_SINGLE_TURN_USER_PROMPT4.format(
                                context=question,
                                response=response_B,
                            )
                        else:
                            user_prompt1 = POINTWISE_SINGLE_TURN_USER_PROMPT7.format(
                                context=question,
                                response=response_A,
                            )
                            user_prompt2 = POINTWISE_SINGLE_TURN_USER_PROMPT7.format(
                                context=question,
                                response=response_B,
                            )

                        messages1 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt1},
                        ]
                        messages2 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt2},
                        ]
                        # prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
                        # prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
                        # # chat template already include special tokens
                        # # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                        # # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                        # # so we need to tokenize without adding special tokens
                        # tokenized_prompt1 = tokenizer(prompt1, add_special_tokens=False, return_length=True)
                        # tokenized_prompt2 = tokenizer(prompt2, add_special_tokens=False, return_length=True)
                        # prompt_ids1 = tokenized_prompt1["input_ids"]
                        # prompt_ids2 = tokenized_prompt2["input_ids"]

                        all_texts.append(messages1)
                        all_texts.append(messages2)
                        # all_prompt_ids.append(prompt_ids1)
                        # all_prompt_ids.append(prompt_ids2)
                        # is_shuffled.append(False)
                        # is_shuffled.append(False)
                    return {
                        "text": all_texts,
                        # "prompt_ids": all_prompt_ids,
                        # "is_shuffled": is_shuffled
                    }

                # if args.mode == "pointwise":
                dataset_prompts = dataset.map(format_judgements_split, batched=True,
                                              fn_kwargs={"optional_chat_template": None},
                                              remove_columns=dataset.column_names)
                prompts = dataset_prompts["text"]
                # print("dataset_prompts", len(dataset_prompts))
                # prompt_ids = dataset_prompts["prompt_ids"]
                # is_shuffled = dataset_prompts["is_shuffled"]
                # else:

                print("prompt:0")
                print(prompts[0])
                answers = [None] * len(prompts)  # Preallocate results list
                done_tasks = 0
                # prompt_ids = dataset_prompts["prompt_ids"]
                # is_shuffled = dataset_prompts["is_shuffled"]
                # ground_truths = dataset_prompts["ground_truths"]

                def update_progress_bar(done, total):
                    # Simple text-based progress bar
                    progress = int(50 * done / total)  # Calculate progress (50 chars width)
                    sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
                    sys.stdout.flush()

                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    # Submit all tasks and hold their futures in a list
                    future_to_index = {executor.submit(chat_completion_gpt5_proxy, args.model, messages): i for
                                       i, messages in enumerate(prompts)}

                    # As tasks complete, update progress and store results in the original order
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        answers[index] = future.result()
                        done_tasks += 1
                        update_progress_bar(done_tasks, len(prompts))
                answers = [[a] for a in answers]

                results = []
                scores = [[process_judgement(sub_a, args.prompt) for sub_a in a] for a in answers]
                error_num = sum([len([s_s for s_s in s if s_s == "error"]) for s in scores])
                print(f"error num: {error_num}/{len(scores) * args.voting}")
                for i in range(0, len(scores), 2):
                    s1 = [s for s in scores[i] if s != "error"]
                    s2 = [s for s in scores[i + 1] if s != "error"]
                    if len(s1) == 0:
                        s1 = "error"
                    else:
                        s1 = sum(s1) / len(s1)
                    if len(s2) == 0:
                        s2 = "error"
                    else:
                        s2 = sum(s2) / len(s2)

                    if s1 != "error" and s2 != "error" and s1 > s2:
                        results.append("A>B")
                        # results.append(1)
                    elif s1 != "error" and s2 != "error" and s1 < s2:
                        results.append("B>A")
                        # results.append(0)
                    elif s1 != "error" and s2 != "error" and s1 == s2:
                        results.append("tie")
                        # results.append(0.5)
                    else:
                        results.append("error")
                        # results.append(0.0)
                tie_num = len([r for r in results if r == "tie"])
                print(f"tie_num: {tie_num}/{len(results)}")
                new_answers = [answers[i * 2][0] + "#@#\n" + answers[i * 2 + 1][0] for i in range(len(results))]
                answers = new_answers
                is_shuffled = [False for _ in range(len(results))]

                for pair, res, ans in zip(pairs, results, answers):
                    pair['judgments'] = [{"decision": res}]
                    pair['answer'] = ans
        else:
            if args.num_gpus > 1:
                # Set the environment variable
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            model = LLM(
                args.model,
                trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=args.num_gpus,
                gpu_memory_utilization=args.vllm_gpu_util,
                max_model_len=args.max_tokens,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            if tokenizer.chat_template is None:
                template_path = os.path.join(args.model, "chat_template.jinja")
                with open(template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
            if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
                stop_token_ids = [128009]
            else:
                stop_token_ids = None
            if args.mode == "pairwise":
                def format_judgements(batch, optional_chat_template=None):
                    all_texts = []
                    all_prompt_ids = []
                    is_shuffled = []
                    for i in range(len(batch["question"])):
                        response_A = batch["response_A"][i]
                        response_B = batch["response_B"][i]
                        question = batch["question"][i]
                        label = batch['label'][i]
                        if "RM-R1" in args.model:
                            system_prompt = RM_R1_INSTRUCT_SYSTEM_PROMPT
                            user_prompt1 = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(
                                question=question,
                                answer_a=response_A,
                                answer_b=response_B,
                            )
                            user_prompt2 = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(
                                question=question,
                                answer_a=response_B,
                                answer_b=response_A,
                            )

                        else:
                            system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.
            
            Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.
            
            When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.
            
            Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.
            
            Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.
            
            After providing your explanation, you must output only one of the following choices as your final verdict with a label:
            
            1. Assistant A is better: [[A>B]] 
            2. Assistant B is better: [[B>A]]
            
            Example output: "My final verdict is Assistant A is better: [[A>B]]"."""
                            user_prompt1 = PAIRWISE_USER_PROMPT.format(
                                context=question,
                                response_a=response_A,
                                response_b=response_B,
                            )
                            user_prompt2 = PAIRWISE_USER_PROMPT.format(
                                context=question,
                                response_a=response_B,
                                response_b=response_A,
                            )

                        messages1 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt1},
                        ]
                        messages2 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt2},
                        ]
                        prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
                        prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
                        tokenized_prompt1 = tokenizer(prompt1, add_special_tokens=False, return_length=True)
                        tokenized_prompt2 = tokenizer(prompt2, add_special_tokens=False, return_length=True)
                        prompt_ids1 = tokenized_prompt1["input_ids"]
                        prompt_ids2 = tokenized_prompt2["input_ids"]

                        all_texts.append(prompt1)
                        all_texts.append(prompt2)
                        all_prompt_ids.append(prompt_ids1)
                        all_prompt_ids.append(prompt_ids2)
                        is_shuffled.append(False)
                        is_shuffled.append(False)
                    return {
                        "text": all_texts,
                        "prompt_ids": all_prompt_ids,
                        "is_shuffled": is_shuffled
                    }
                dataset_prompts = dataset.map(format_judgements, batched=True,
                                              fn_kwargs={"optional_chat_template": None},
                                              remove_columns=dataset.column_names)
                prompts = dataset_prompts["text"]
                prompt_ids = dataset_prompts["prompt_ids"]
                is_shuffled = dataset_prompts["is_shuffled"]

                print("prompt:0")
                print(prompts[0])
                if args.voting == 1:
                    sampling_params = SamplingParams(
                        n=args.voting,
                        temperature=0,
                        top_p=1,
                        max_tokens=args.max_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                else:
                    sampling_params = SamplingParams(
                        n=args.voting,
                        temperature=1.0,
                        top_p=1,
                        max_tokens=args.max_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                outputs = model.generate(prompts, sampling_params=sampling_params)
                answers = [o.outputs[0].text for o in outputs]
                results = []
                scores = [process_judgement(a, f"{args.mode}-RM-R1" if args.mode == "pairwise" and "RM-R1" in args.model else args.mode) for a in answers]
                error_num = len([s for s in scores if s == "error"])
                print(f"error num: {error_num}/{len(scores)}")
                for i in range(0, len(scores), 2):
                    s1, s2 = scores[i], scores[i + 1]
                    results.append([{"decision": s1}, {"decision": s2}])
                new_answers = [answers[i * 2] + "#@#\n" + answers[i * 2 + 1] for i in range(len(results))]
                answers = new_answers

                for pair, res, ans in zip(pairs, results, answers):
                    pair['judgments'] = res
                    pair['answer'] = ans

            elif args.mode == "pointwise":
                def format_judgements_split(batch, optional_chat_template=None):
                    # TODO expand this to include fastchat chat templates if needed

                    all_texts = []
                    all_prompt_ids = []
                    is_shuffled = []
                    for i in range(len(batch["question"])):
                        response_A = batch["response_A"][i]
                        response_B = batch["response_B"][i]
                        question = batch["question"][i]
                        label = batch['label'][i]
                        if "llama" in args.model:
                            system_prompt = ""
                        else:
                            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

                        if args.prompt == "prompt4":
                            user_prompt1 = POINTWISE_SINGLE_TURN_USER_PROMPT4.format(
                                context=question,
                                response=response_A,
                            )
                            user_prompt2 = POINTWISE_SINGLE_TURN_USER_PROMPT4.format(
                                context=question,
                                response=response_B,
                            )
                        else:
                            user_prompt1 = POINTWISE_SINGLE_TURN_USER_PROMPT7.format(
                                context=question,
                                response=response_A,
                            )
                            user_prompt2 = POINTWISE_SINGLE_TURN_USER_PROMPT7.format(
                                context=question,
                                response=response_B,
                            )

                        messages1 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt1},
                        ]
                        messages2 = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_prompt2},
                        ]
                        prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
                        prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
                        # chat template already include special tokens
                        # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                        # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                        # so we need to tokenize without adding special tokens
                        tokenized_prompt1 = tokenizer(prompt1, add_special_tokens=False, return_length=True)
                        tokenized_prompt2 = tokenizer(prompt2, add_special_tokens=False, return_length=True)
                        prompt_ids1 = tokenized_prompt1["input_ids"]
                        prompt_ids2 = tokenized_prompt2["input_ids"]

                        all_texts.append(prompt1)
                        all_texts.append(prompt2)
                        all_prompt_ids.append(prompt_ids1)
                        all_prompt_ids.append(prompt_ids2)
                        is_shuffled.append(False)
                        is_shuffled.append(False)
                    return {
                        "text": all_texts,
                        "prompt_ids": all_prompt_ids,
                        "is_shuffled": is_shuffled
                    }
                # if args.mode == "pointwise":
                dataset_prompts = dataset.map(format_judgements_split, batched=True,
                                              fn_kwargs={"optional_chat_template": None},
                                              remove_columns=dataset.column_names)
                prompts = dataset_prompts["text"]
                prompt_ids = dataset_prompts["prompt_ids"]
                is_shuffled = dataset_prompts["is_shuffled"]
            # else:

                print("prompt:0")
                print(prompts[0])
                if args.voting == 1:
                    sampling_params = SamplingParams(
                        n=args.voting,
                        temperature=0,
                        top_p=1,
                        max_tokens=args.max_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                else:
                    sampling_params = SamplingParams(
                        n=args.voting,
                        temperature=1.0,
                        top_p=1,
                        max_tokens=args.max_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                outputs = model.generate(prompts, sampling_params=sampling_params)

                if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
                    answers = outputs
                else:
                    if args.mode == "pointwise":
                        answers = [[o.outputs[i].text for i in range(args.voting)] for o in outputs]
                    else:
                        answers = [o.outputs[0].text for o in outputs]

                if args.mode == 'pointwise':
                    results = []
                    scores = [[process_judgement(sub_a, args.prompt) for sub_a in a] for a in answers]
                    error_num = sum([len([s_s for s_s in s if s_s == "error"]) for s in scores])
                    print(f"error num: {error_num}/{len(scores)*args.voting}")
                    for i in range(0, len(scores), 2):
                        s1 = [s for s in scores[i] if s != "error"]
                        s2 = [s for s in scores[i+1] if s != "error"]
                        if len(s1)==0:
                            s1 = "error"
                        else:
                            s1 = sum(s1) / len(s1)
                        if len(s2)==0:
                            s2 = "error"
                        else:
                            s2 = sum(s2) / len(s2)

                        if s1 != "error" and s2 != "error" and s1 > s2:
                            results.append("A>B")
                            # results.append(1)
                        elif s1 != "error" and s2 != "error" and s1 < s2:
                            results.append("B>A")
                            # results.append(0)
                        elif s1 != "error" and s2 != "error" and s1 == s2:
                            results.append("tie")
                            # results.append(0.5)
                        else:
                            results.append("error")
                            # results.append(0.0)
                    tie_num = len([r for r in results if r == "tie"])
                    print(f"tie_num: {tie_num}/{len(results)}")
                    new_answers = [answers[i * 2][0] + "#@#\n" + answers[i * 2 + 1][0] for i in range(len(results))]
                    answers = new_answers
                    is_shuffled = [False for _ in range(len(results))]

                    for pair, res, ans in zip(pairs, results, answers):
                        pair['judgments'] = [{"decision": res}]
                        pair['answer'] = ans

        # 统计失败的 pair
        failed_pairs = []
        for pair in pairs:
            if not pair.get("judgments") or len(pair["judgments"]) == 0:
                failed_pairs.append(pair["pair_id"])
            elif pair["judgments"][0] is None:
                failed_pairs.append(pair["pair_id"])
            elif isinstance(pair["judgments"][0], dict) and pair["judgments"][0].get("decision") is None:
                failed_pairs.append(pair["pair_id"])

        for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
            score = metrics.compute_final_metrics(pairs, args.mode == "pairwise", include_fn=lambda x: x["source"].startswith(source))
            print(f"{source if source else 'Overall'}: {score:.2f}%.")

        output_dir = os.path.join(args.meta_result_save_dir, args.model_save_name, "judge")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.mode}_{mm}.jsonl")
        print('output path:', output_path)
        file_operations.write_to_jsonl(output_path, pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--judge_name', type=str, required=True) # name of judge, should correspond to an entry in utils/judges/get_judge_from_judge_name_and_model.
    parser.add_argument('--model', type=str, required=True) # model to be used by judge.
    # parser.add_argument('--single_game', action="store_true") # by default, we run each pair through twice (A,B) and (B,A). This flag will only run the original ordering, and should be used if a judge is order-independent.
    parser.add_argument('--seed', type=int, default=42) # seed to use.
    # parser.add_argument('--pairs', type=str, required=True) # path to jsonl containing pairs for judging
    parser.add_argument('--device_map', type=str, default=None) # device map for multi-GPU: "auto" for automatic, or number like "4" for 4 GPUs, or custom dict string
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    parser.add_argument("--max_tokens", type=int, default=8192, help="max tokens for vllm")
    parser.add_argument("--model_save_name", default="default_save", type=str)
    parser.add_argument("--mode", default="pairwise", type=str)
    parser.add_argument("--meta_result_save_dir", default="./result", type=str)
    parser.add_argument("--subset", default="gpt", type=str)
    parser.add_argument("--voting", default=1, type=int)
    parser.add_argument("--prompt", default="prompt4", type=str)
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    args = parser.parse_args()
    # args.single_game = (args.mode == "pairwise")
    main(args)