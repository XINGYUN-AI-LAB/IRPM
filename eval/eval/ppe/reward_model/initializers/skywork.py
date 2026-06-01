from utils.core import register
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import bfloat16


@register("Skywork/Skywork-Reward-Gemma-2-27B", init_registry)
@register("Skywork/Skywork-Reward-Llama-3.1-8B", init_registry)
@register("skywork", init_registry)
class SkyworkRewardModelHandler(BaseRewardModelHandler):

    def __init__(self, model_name=None) -> None:
        super().__init__(model_name=model_name)

    def init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        return model

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def init_pipeline(self):
        return pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=pipeline_registry.get("SkyworkPipeline"),
        )
