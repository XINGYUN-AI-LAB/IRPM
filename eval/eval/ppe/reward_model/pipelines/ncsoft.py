from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from typing import Dict
import torch
from transformers import TextClassificationPipeline


@register("NCSOFTPipeline", pipeline_registry)
class NCSOFTPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        config = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1,
        }
        kwargs.update(config)
        super().__init__(**kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:

        formatted = self.tokenizer.apply_chat_template(
            inputs, tokenize=False, add_generation_prompt=False
        ).replace(self.tokenizer.bos_token, "")

        return super().preprocess(formatted, **tokenizer_kwargs)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        output = model_outputs["logits"][0]

        return output.cpu().float().item()
