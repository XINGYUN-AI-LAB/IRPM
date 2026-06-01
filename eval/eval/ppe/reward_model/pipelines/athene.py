from utils.core import register
from reward_model.pipelines.base import BaseRewardPipeline, pipeline_registry
from typing import Dict
import torch


@register("AtheneRewardPipeline", pipeline_registry)
class AtheneRewardPipeline(BaseRewardPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        return_tensors = self.framework

        formatted = self.tokenizer.apply_chat_template(inputs, tokenize=False)

        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
        )
