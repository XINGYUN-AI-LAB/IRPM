# coding:utf-8
from ast import main
import sys
import os
from pathlib import Path

parent_dir = str(Path(__file__).parent.resolve().parent)
sys.path.append(parent_dir)

import json
import argparse
# from utils.data_util import read_json_from_file, write_filename
# from utils.logging_util import logger

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoModel, AutoTokenizer, AutoConfig
import time
import subprocess

class EvaluateProject():
    def __init__(self, args) -> None:
        self.device = "cuda:0"
        # self.eval_file = args.eval_file
        self.model_path = args.model
        # self.write_file = args.write_file
        # self.time = int(time.time())
        # self.cutoff = int(int(args.cutoff) * 1.57)
        self.model, self.tokenizer = self.load_eval_model(self.model_path)

    def load_eval_data(self, filename):
        data = read_json_from_file(filename)
        logger.info(f"eval data len is {len(data)} ")
        return data

    def load_eval_model(self, model_path):
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "sliding_window"):
            config.sliding_window = None
        if hasattr(config, "use_sliding_window"):
            config.use_sliding_window = False
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                     device_map='cpu', config=config)
        model = AutoModelForCausalLMWithValueHead(model)

        with safe_open(model_path + "/value_head.safetensors", framework="pt", device='cpu') as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}

        model.load_state_dict(vhead_params, strict=False)
        print("vhead load success")

        model = model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
        model.is_peft_model = False
        model.eval()
        return model, tokenizer

    def pipline2(self, eval_data):

        outputs = []
        for i, text in enumerate(eval_data):
            # out = eval_data[i]
            # if out['instruction'] == None or out['instruction'] == '无。':
            #     continue
            # truncated_history = extract_last_n_conversations_content_only(out["instruction"], n=1)
            # user_input = out['user_query']
            # torch.cuda.empty_cache()
            # messages = [{"role": "user", "content": truncated_history + "\n" + user_input},
            #             {"role": "assistant", "content": out['chosen']}]
            # text = self.tokenizer.apply_chat_template(
            #     message,
            #     tokenize=False,
            #     add_generation_prompt=False,
            #     enable_thinking=False
            # )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                _, _, values = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            last_idx = attention_mask[0].nonzero()[-1].item()
            outputs.append(values[:, last_idx].item())

        print(len(eval_data), len(outputs))
        # self.save_file_to_local_oss(json.dumps(outputs, ensure_ascii=False, indent=3))
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eval_file", type=str, default="", help="oss path")
    parser.add_argument("--model_path", type=str, default="", help="oss path")
    parser.add_argument("--write_file", type=str, default="", help="local path")
    parser.add_argument("--device", type=str, default="0", )
    parser.add_argument("--cutoff", type=str, default="1000000", )

    args = parser.parse_args()
    logger.info(f"[args]: {args}")

    evaluateProject = EvaluateProject(args)
    evaluateProject.pipline()

