from utils.core import register
from reward_model.models.base import model_registry
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, pipeline
from torch import bfloat16


@register("Nexusflow/Starling-RM-34B", init_registry)
@register("starling-beta", init_registry)
class StarlingRewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__()

    def init_model(self):
        model_cls = model_registry.get("Nexusflow/Starling-RM-34B")
        return model_cls.from_pretrained(
            "Nexusflow/Starling-RM-34B", torch_dtype=bfloat16
        )

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
        tokenizer.truncation_side = "left"
        tokenizer.pad_token_id = 0

        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("BaseRewardPipeline"),
        )
