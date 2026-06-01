from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import re

import utils.prompts as prompts
import utils.models as models

# run judge on pairs
# judges should take a question and two responses, and return a decision (e.g., A>B or B>A)
# Note B>A and not A<B
# Judge.get_judgment must be async!
# For a new judge, add a corresponding entry to get_judge_from_judge_name_and_model


class Judge(ABC):
    @abstractmethod
    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        pass


class ArenaHard(Judge):
    # Implementation follows 
    # https://github.com/lmarena/arena-hard-auto/blob/4ce0f0087776158a4461162cbef1d9bb5464bb57/gen_judgment.py

    def __init__(self, model_name):
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.number_of_judgment_attempts = 2

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        system_message = prompts.render_template(
            "arena_hard_judge_system")
        user_message = prompts.render_template("arena_hard_judge_prompt",
                                                    prompt=question, answer_a=answer_A, answer_b=answer_B)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        judgment = ""
        for _ in range(self.number_of_judgment_attempts):
            new_judgment = await self.api.chat(
                messages=messages,
                temperature=0,
                max_tokens=4096,
            )
            judgment += ("\n" + new_judgment)
            score, try_again = self.get_score(
                judgment, re.compile("\[\[([AB<>=]+)\]\]"))
            messages.append({"role": "assistant", "content": new_judgment})
            if not try_again:
                break
            messages.append(
                {"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": judgment
            },
            "decision": score.replace(">>", ">").strip() if score else None
        }

    def get_score(cls, judgment: str, pattern: str, pairwise: bool = True) -> Tuple[Union[int, str], Optional[bool]]:
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0])
        else:
            return None, False
        
class Vanilla(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pairwise_result(self, raw_output):
        print("raw:", raw_output)
        if raw_output == "Output (a)":
            return "A>B"
        elif raw_output == "Output (b)":
            return "B>A"
        raise Exception("Cannot parse output:", raw_output)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "vanilla_prompt", question=question, answer_a=answer_A, answer_b=answer_B)
        print("prompt:", prompt)
        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )

        pred_label = self.extract_pairwise_result(output.strip())

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label
        }

class PandaLM(Judge):

    def __init__(self, model_name) -> None:
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, use_fast=False)
        
    def truncate_responses(self, question, answer_A, answer_B, context_limit, max_new_tokens, truncation_side):
        template_with_question = prompts.render_template("pandalm_prompt", instruction=question, resp1="", resp2="")
        len_template = len(self.tokenizer(template_with_question).input_ids) # includes special BOS token <s>
        tokens_per_response = (context_limit - max_new_tokens - len_template) // 2 - 2 # each response should be truncated to a length of tokens_per_response
        
        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = answer_A_tokenized[:tokens_per_response] if truncation_side == "right" else answer_A_tokenized[-tokens_per_response:] # left
        answer_A_truncated = self.tokenizer.decode(answer_A_tokenized_truncated) # should not be any special tokens anyways
        
        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = answer_B_tokenized[:tokens_per_response] if truncation_side == "right" else answer_B_tokenized[-tokens_per_response:] # left
        answer_B_truncated = self.tokenizer.decode(answer_B_tokenized_truncated) # should not be any special tokens anyways
        
        return answer_A_truncated, answer_B_truncated

    def build_pandalm_prompt(self, instruction, resp1, resp2):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        input_sequence = prompts.render_template(
            "pandalm_prompt", instruction=instruction, resp1=resp1, resp2=resp2)
        return input_sequence + "\n" # why does jinja strip the training new line?

    def parse_pandalm_response(self, text):
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0

    def postprocess_output(self, text):
        text = text.strip()
        self.pattern.sub("", text.strip()).strip()
        return text

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=150, # we only need the first few tokens to determine decision
            truncation_side="left"
        )

        prompt = self.build_pandalm_prompt(
            instruction=question,
            resp1=answer_A,
            resp2=answer_B,
        )

        output = await self.api.complete(
            prompt=prompt,
            temperature=0,
            top_p=1,
            max_tokens=150,
            extra_body={
                "use_beam_search": True,
                "best_of": 4,
                "early_stopping": True,
                "repetition_penalty": 1.2,
            },
        )

        resp = self.postprocess_output(output)
        out = self.parse_pandalm_response(resp)
        if out == 1:
            decision = "A>B"
        elif out == 2:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": resp,
            },
            "decision": decision
        }


class JudgeLM(Judge):
    from transformers import AutoTokenizer
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, use_fast=False)
        
    def truncate_responses(self, question, answer_A, answer_B, context_limit, max_new_tokens, truncation_side):
        template_with_question = prompts.render_template("judgelm_prompt", question=question, answer_1="", answer_2="")
        len_template = len(self.tokenizer(template_with_question).input_ids) # includes special BOS token <s>
        tokens_per_response = (context_limit - max_new_tokens - len_template) // 2 - 2 # each response should be truncated to a length of tokens_per_response
        
        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = answer_A_tokenized[:tokens_per_response] if truncation_side == "right" else answer_A_tokenized[-tokens_per_response:] # left
        answer_A_truncated = self.tokenizer.decode(answer_A_tokenized_truncated) # should not be any special tokens anyways
        
        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = answer_B_tokenized[:tokens_per_response] if truncation_side == "right" else answer_B_tokenized[-tokens_per_response:] # left
        answer_B_truncated = self.tokenizer.decode(answer_B_tokenized_truncated) # should not be any special tokens anyways
        
        return answer_A_truncated, answer_B_truncated

    def parse_score(self, review):
        try:
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                raise Exception()
        except Exception:
            return [-1, -1]

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=16,
            truncation_side="right"
        )
        
        prompt = prompts.render_template(
            "judgelm_prompt", question=question, answer_1=answer_A, answer_2=answer_B)

        output = await self.api.complete(
            prompt=prompt,
            temperature=0.0, # https://github.com/baaivision/JudgeLM/blob/ce12b12779764fe06e28c797cecee86018a298e4/judgelm/llm_judge/gen_model_judgement_multi.py#L235
            max_tokens=16,
        )

        scores = self.parse_score(output)

        if scores[0] > scores[1]:
            decision = "A>B"
        elif scores[0] < scores[1]:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }


class AutoJ(Judge):

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pariwise_result(self, raw_output):
        raw_output = raw_output.strip()
        pos = raw_output.rfind('final decision is ')
        pred_label = None
        if pos != -1:
            pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
            if pred_rest.startswith('response 1'):
                pred_label = "A>B"
            elif pred_rest.startswith('response 2'):
                pred_label = "B>A"
            elif pred_rest.startswith('tie'):
                pred_label = "A=B"
        return pred_label

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "autoj_prompt", question=question, response=answer_A, response_another=answer_B)

        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )  # SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024) https://github.com/GAIR-NLP/auto-j

        pred_label = self.extract_pariwise_result(output)

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label
        }


class Prometheus2(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.rubric = "[Are the model's responses factually correct and well-supported by evidence?]" # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py
        self.REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def _parse_output_relative(self, output):
        explicit_pattern = r"""
            (?:                                # Start of non-capturing group
                \[RESULT\]|\[RESULT:\s*|        # Match [RESULT] or [RESULT:
                \[Response\s+|                  # Match [Response
                # Match [Result] or [Result] Response
                \[Result\](?:\s+Response)?|
                \[Result:\s*|                   # Match [Result:
                # Match Result: at the start of a line
                (?:^|\n)Result:?\s*
            )                                   # End of non-capturing group
            \s*                                 # Allow any whitespace
            (A|B)                               # Capture A or B
            (?:\]|\s|$)                         # Allow closing bracket, whitespace, or end of string
        """
        match = re.search(
            explicit_pattern, output, re.IGNORECASE | re.VERBOSE | re.MULTILINE
        )

        if match:
            result = match.group(1).upper()
            feedback = output[: match.start()].strip()
            return output, result

        return None, None

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "prometheus2_prompt",
            instruction=question,
            response_A=answer_A,
            response_B=answer_B,
            rubric=self.rubric,
        )

        messages = [
            {"role": "system", "content": self.REL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        ) # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/utils.py
        
        _, scores = self._parse_output_relative(output)
        
        decision = None # no tie option 
        if scores == "A":
            decision = "A>B"
        elif scores == "B":
            decision = "B>A"
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }
        

class SkyworkCritic(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "skywork_critic_prompt",
            input=question,
            response_a=answer_A,
            response_b=answer_B,
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        if "A" in output:
            decision = "A>B"
        elif "B" in output:
            decision = "B>A"
        else:
            decision = None
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }
        
        
class InternLM2Reward(Judge):
    def __init__(self, model_name="internlm/internlm2-20b-reward", device="cuda:0"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = model_name
        self.device = device
        self.rm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_A}]
        conv2 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_B}]

        score1 = self.rm.get_score(self.rm_tokenizer, conv1)
        score2 = self.rm.get_score(self.rm_tokenizer, conv2)
        
        judgement = 'A>B' if score1 > score2 else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score1, score2]
            },
            "decision": judgement
        }
        

class GRMReward(Judge):
    def __init__(self, model_name="Ray2333/GRM-Gemma-2B-rewardmodel-ft", device="cuda:0"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
        ).to(self.device)

    def get_reward(self, message):
        import torch
        message_template = self.tokenizer.apply_chat_template(message, tokenize=False)
        
        kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(message_template, **kwargs)
        
        with torch.no_grad():
            reward_tensor = self.reward_model(
                tokens["input_ids"][0].view(1,-1).to(self.device), 
                attention_mask=tokens["attention_mask"][0].view(1,-1).to(self.device)
            )[0]
            reward = reward_tensor.cpu().detach().item()
        
        return reward

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        message_A = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_A}
        ]
        message_B = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_B}
        ]

        score_A = self.get_reward(message_A)
        score_B = self.get_reward(message_B)

        judgement = 'A>B' if score_A > score_B else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score_A, score_B]
            },
            "decision": judgement
        }


class SkyworkReward(Judge):
    def __init__(self, model_name, device="cuda:0"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_name = model_name
        self.device = device
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        import torch
        conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_A}]
        conv2 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_B}]

        conv1_formatted = self.rm_tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = self.rm_tokenizer.apply_chat_template(conv2, tokenize=False)
        conv1_tokenized = self.rm_tokenizer(conv1_formatted, return_tensors="pt").to(self.device)
        conv2_tokenized = self.rm_tokenizer(conv2_formatted, return_tensors="pt").to(self.device)

        # Get the reward scores
        with torch.no_grad():
            score1 = self.rm(**conv1_tokenized).logits[0][0].item()
            score2 = self.rm(**conv2_tokenized).logits[0][0].item()

        judgement = 'A>B' if score1 > score2 else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score1, score2]
            },
            "decision": judgement
        }
        

class CompassJudger(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        
    def get_score(cls, judgment: str, pattern: str, pairwise: bool = True) -> Tuple[Union[int, str], Optional[bool]]:
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0])
        else:
            return None, False
        
    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        system_message = prompts.render_template(
            "arena_hard_judge_system")
        user_message = prompts.render_template("arena_hard_judge_prompt",
                                                    prompt=question, answer_a=answer_A, answer_b=answer_B)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        score, _ = self.get_score(output, re.compile("\[\[([AB<>=]+)\]\]"))
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": output,
            },
            "decision": score.replace(">>", ">").strip() if score else None
        }


class CustomRewardModel(Judge):
    """通用 Reward Model 类，基于 GRMReward 结构，支持生成式和分类式模型"""
    def __init__(self, model_name, device="cuda:0", device_map=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
        
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        
        # 如果指定了 device_map，使用多卡自动分配
        # 否则使用单卡
        if device_map is None:
            device_map = device
        
        # 首先尝试作为生成式模型加载
        try:
            print(f"Attempting to load {model_name} as AutoModelForCausalLM (generative)...")
            if device_map == "auto" or isinstance(device_map, dict):
                # 多卡模式：使用 balanced 策略避免 tensor parallel
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    # 使用 "balanced" 或 "balanced_low_0" 来避免触发 tensor parallel
                    # 这些策略会进行模型并行而不是 tensor parallel
                    # 尝试禁用 flash attention 以避免多卡模型并行时的形状错误
                    try:
                        self.reward_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            device_map="balanced_low_0",  # 使用 balanced_low_0 避免 tensor parallel
                            low_cpu_mem_usage=True,
                            attn_implementation="eager",  # 禁用 flash attention，使用 eager attention
                        )
                    except TypeError:
                        # 如果模型不支持 attn_implementation 参数，不使用它
                        self.reward_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            device_map="balanced_low_0",
                            low_cpu_mem_usage=True,
                        )
                else:
                    # 单卡情况
                    self.reward_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    ).to(self.device)
                # tokenizer 放在 CPU 或第一个 GPU
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                # 输入数据放在第一个 GPU
                self.device = "cuda:0"
            else:
                # 单卡模式
                self.reward_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_type = "generative"
            print(f"Successfully loaded as generative model on device(s): {device_map if device_map else device}")
        except Exception as e1:
            error_msg = str(e1)
            # 检查是否是模型架构不支持的错误
            if "does not recognize this architecture" in error_msg or "model type" in error_msg.lower():
                print(f"\n❌ Error: Transformers library does not support this model architecture.")
                print(f"   Error details: {error_msg}")
                print(f"\n💡 Solutions:")
                print(f"   1. Update transformers: pip install --upgrade transformers")
                print(f"   2. Install from source: pip install git+https://github.com/huggingface/transformers.git")
                print(f"   3. Check if the model path is correct: {model_name}")
                raise ValueError(f"Model architecture not supported by current transformers version. Please update transformers or check model path.")
            # 如果失败，尝试作为分类模型加载（类似 GRMReward）
            print(f"Failed as generative: {error_msg}")
            print(f"Trying AutoModelForSequenceClassification...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if device_map == "auto" or isinstance(device_map, dict):
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 1:
                        try:
                            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                                model_name, 
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="balanced_low_0",
                                low_cpu_mem_usage=True,
                                attn_implementation="eager",  # 禁用 flash attention
                            )
                        except TypeError:
                            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                                model_name, 
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="balanced_low_0",
                                low_cpu_mem_usage=True,
                            )
                    else:
                        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                            model_name, 
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                        ).to(self.device)
                    self.device = "cuda:0"
                else:
                    self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    ).to(self.device)
                self.model_type = "classification"
                print(f"Successfully loaded as classification model on device(s): {device_map if device_map else device}")
            except Exception as e2:
                error_msg = str(e2)
                # 检查是否是模型架构不支持的错误
                if "does not recognize this architecture" in error_msg or "model type" in error_msg.lower():
                    print(f"\n❌ Error: Transformers library does not support this model architecture.")
                    print(f"   Error details: {error_msg}")
                    print(f"\n💡 Solutions:")
                    print(f"   1. Update transformers: pip install --upgrade transformers")
                    print(f"   2. Install from source: pip install git+https://github.com/huggingface/transformers.git")
                    print(f"   3. Check if the model path is correct: {model_name}")
                    raise ValueError(f"Model architecture not supported by current transformers version. Please update transformers or check model path.")
                # 最后尝试 AutoModel（InternLM2 类型）
                print(f"Failed as classification: {error_msg}")
                print(f"Trying AutoModel...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if device_map == "auto" or isinstance(device_map, dict):
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 1:
                        try:
                            self.reward_model = AutoModel.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="balanced_low_0",
                                low_cpu_mem_usage=True,
                                attn_implementation="eager",  # 禁用 flash attention
                            )
                        except TypeError:
                            self.reward_model = AutoModel.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="balanced_low_0",
                                low_cpu_mem_usage=True,
                            )
                    else:
                        self.reward_model = AutoModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                        ).to(self.device)
                    self.device = "cuda:0"
                else:
                    self.reward_model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    ).to(self.device)
                self.model_type = "auto_model"
                print(f"Successfully loaded as AutoModel on device(s): {device_map if device_map else device}")
            except Exception as e3:
                error_msg = str(e3)
                # 检查是否是模型架构不支持的错误
                if "does not recognize this architecture" in error_msg or "model type" in error_msg.lower():
                    print(f"\n❌ Error: Transformers library does not support this model architecture.")
                    print(f"   Error details: {error_msg}")
                    print(f"\n💡 Solutions:")
                    print(f"   1. Update transformers: pip install --upgrade transformers")
                    print(f"   2. Install from source: pip install git+https://github.com/huggingface/transformers.git")
                    print(f"   3. Check if the model path is correct: {model_name}")
                    raise ValueError(f"Model architecture not supported by current transformers version. Please update transformers or check model path.")
                else:
                    # 其他错误，重新抛出
                    raise Exception(f"Failed to load model from {model_name} as generative, classification, or auto model. Last error: {error_msg}")

    def get_reward(self, message):
        """类似 GRMReward 的 get_reward 方法，用于分类式模型"""
        import torch
        message_template = self.tokenizer.apply_chat_template(message, tokenize=False)
        
        kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(message_template, **kwargs)
        
        with torch.no_grad():
            reward_tensor = self.reward_model(
                tokens["input_ids"][0].view(1,-1).to(self.device), 
                attention_mask=tokens["attention_mask"][0].view(1,-1).to(self.device)
            )
            # 处理不同的输出格式
            if hasattr(reward_tensor, 'logits'):
                reward = reward_tensor.logits[0][0].item()
            elif isinstance(reward_tensor, tuple):
                reward = reward_tensor[0][0][0].item()
            elif hasattr(self.reward_model, 'get_score'):
                # InternLM2 类型
                return self.reward_model.get_score(self.tokenizer, message)
            else:
                reward = reward_tensor[0].cpu().detach().item()
        
        return reward

    def _generate_sync(self, prompt: str) -> str:
        """同步生成方法，在线程池中执行"""
        import torch
        
        # 确定输入应该放在哪个设备
        # 对于多卡模型并行，输入应该放在第一个设备（通常是 cuda:0）
        input_device = self.device
        
        # 如果模型使用了 device_map，找到第一个设备
        if hasattr(self.reward_model, 'hf_device_map') and self.reward_model.hf_device_map:
            # 找到第一个设备（通常是 embedding 层所在的设备）
            first_device = list(self.reward_model.hf_device_map.values())[0]
            if isinstance(first_device, (list, tuple)):
                first_device = first_device[0]
            input_device = first_device
        
        try:
            # Tokenize 和 Generate
            # 减小 max_length 避免 flash attention 错误
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
            
            # 将输入放到正确的设备
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            
            # Generate with standard parameters
            with torch.no_grad():
                outputs = self.reward_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            error_msg = str(e)
            # 如果是形状错误，尝试更短的输入
            if "shape" in error_msg.lower() or "q must have shape" in error_msg:
                try:
                    # 进一步缩短输入
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    inputs = {k: v.to(input_device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.reward_model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    return generated_text
                except Exception as e2:
                    raise Exception(f"Failed to generate even with shorter input: {e2}")
            else:
                raise

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        import asyncio
        
        # 如果是生成式模型
        if self.model_type == "generative":
            # 构建提示词
            try:
                messages = [
                    {"role": "user", "content": f"Question: {question}\n\nAnswer A: {answer_A}\n\nAnswer B: {answer_B}\n\nWhich answer is better? Please respond with 'A' or 'B'."}
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"Question: {question}\n\nAnswer A: {answer_A}\n\nAnswer B: {answer_B}\n\nWhich answer is better? Please respond with 'A' or 'B'."
            
            # 在线程池中执行同步生成操作，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(None, self._generate_sync, prompt)
            
            # 解析决策
            generated_text_upper = generated_text.strip().upper()
            if "A" in generated_text_upper and "B" in generated_text_upper:
                a_pos = generated_text_upper.find("A")
                b_pos = generated_text_upper.find("B")
                decision = "A>B" if a_pos < b_pos else "B>A"
            elif "A" in generated_text_upper:
                decision = "A>B"
            elif "B" in generated_text_upper:
                decision = "B>A"
            else:
                decision = None
                print(f"Warning: Could not parse decision from: {generated_text}")
            
            return {
                "judgment": {
                    "judge_model": self.model_name,
                    "prompt": prompt,
                    "response": generated_text,
                },
                "decision": decision
            }
        
        # 如果是分类式模型（类似 GRMReward）
        message_A = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_A}
        ]
        message_B = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_B}
        ]

        # 在线程池中执行同步操作，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        score_A, score_B = await asyncio.gather(
            loop.run_in_executor(None, self.get_reward, message_A),
            loop.run_in_executor(None, self.get_reward, message_B)
        )

        judgement = 'A>B' if score_A > score_B else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score_A, score_B]
            },
            "decision": judgement
        }


def get_judge_from_judge_name_and_model(judge_name: str, judge_model: str, device_map=None) -> Judge:
    if judge_name == "arena_hard":
        return ArenaHard(judge_model)
    elif judge_name == "vanilla":
        return Vanilla(judge_model)
    elif judge_name == "panda_lm":
        return PandaLM(judge_model) 
    elif judge_name == "judge_lm":
        return JudgeLM(judge_model)
    elif judge_name == "auto_j":
        return AutoJ(judge_model)
    elif judge_name == "prometheus_2":
        return Prometheus2(judge_model)
    elif judge_name == "skywork_critic":
        return SkyworkCritic(judge_model)
    elif judge_name == "compass_judger":
        return CompassJudger(judge_model)
    elif judge_name == "reward_model":
        if judge_model in ["internlm/internlm2-7b-reward", "internlm/internlm2-20b-reward"]:
            return InternLM2Reward(judge_model)
        elif judge_model in ["Ray2333/GRM-Gemma-2B-rewardmodel-ft"]:
            return GRMReward(judge_model)
        elif judge_model in ["Skywork/Skywork-Reward-Gemma-2-27B", "Skywork/Skywork-Reward-Llama-3.1-8B"]:
            return SkyworkReward(judge_model)
        else:
            # 对于其他模型（包括本地路径），使用通用的 CustomRewardModel
            return CustomRewardModel(judge_model, device_map=device_map)
    else:
        raise NotImplementedError(
            f"Judge with name {judge_name} for model with name {judge_model} is not yet implemented.")