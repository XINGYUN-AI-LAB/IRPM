from utils.core import register
from reward_model.models.base import model_registry
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, AutoConfig, pipeline


@register("NaiveVerbosityModel", init_registry)
class NaiveVerbosityModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__()

    def init_model(self):
        model = model_registry.get("NaiveVerbosityModel")(
            config=AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        model.PAD_ID = self.tokenizer.pad_token_id
        return model

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
        return tokenizer

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("BaseRewardPipeline"),
        )
