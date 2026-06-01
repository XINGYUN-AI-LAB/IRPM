from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, AutoModel, pipeline
from torch import float16


@register("internlm/internlm2-20b-reward", init_registry)
@register("internlm/internlm2-7b-reward", init_registry)
@register("internlm/internlm2-1_8b-reward", init_registry)
@register("internlm", init_registry)
class InternLM2RewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__(model_name=model_name)

    def init_model(self):
        model = AutoModel.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=float16,
            trust_remote_code=True,
        )

        return model

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("InternLM2Pipeline"),
        )
