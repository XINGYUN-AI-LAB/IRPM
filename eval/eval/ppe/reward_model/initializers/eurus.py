from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, AutoModel, pipeline


@register("openbmb/Eurus-RM-7b", init_registry)
@register("eurus", init_registry)
class EurusRewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__(model_name=model_name)

    def init_model(self):
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        return model

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        tokenizer.pad_token_id = 0
        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("EurusRMPipeline"),
        )
