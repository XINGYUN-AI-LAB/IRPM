from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


@register("RLHFlow/ArmoRM-Llama3-8B-v0.1", init_registry)
@register("armo", init_registry)
class ArmoRewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__(model_name=model_name)

    def init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        return model

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True
        )
        tokenizer.truncation_side = "left"
        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("ArmoRMPipeline"),
        )
