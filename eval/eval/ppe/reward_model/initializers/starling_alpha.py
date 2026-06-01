from utils.core import register
from reward_model.models.base import model_registry
from reward_model.pipelines.base import pipeline_registry
from reward_model.initializers.base import BaseRewardModelHandler, init_registry
from transformers import AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
import os
import torch


@register("berkeley-nest/Starling-RM-7B-alpha", init_registry)
@register("starling-alpha", init_registry)
class StarlingAlphaRewardModelHandler(BaseRewardModelHandler):
    def __init__(self, model_name=None) -> None:
        super().__init__()

    def init_model(self):

        # I'm so sorry for this...
        reward_model = model_registry.get("berkeley-nest/Starling-RM-7B-alpha")(
            "meta-llama/Llama-2-7b-chat-hf"
        )
        directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        reward_model.eval().requires_grad_(False)

        return reward_model

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
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
