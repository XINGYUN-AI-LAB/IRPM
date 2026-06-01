from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, pipeline
from torch import bfloat16


@register("NCSOFT/Llama-3-OffsetBias-RM-8B", init_registry)
@register("ncsoft", init_registry)
class NCSOFTRewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__(model_name=model_name)

    def init_model(self):
        pass

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": bfloat16},
            pipeline_class=pipeline_registry.get("NCSOFTPipeline"),
        )
