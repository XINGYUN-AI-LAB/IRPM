from transformers import TextClassificationPipeline
import torch
from typing import Dict


class BaseRewardPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        return_tensors = self.framework

        formatted = self.tokenizer.apply_chat_template(inputs, tokenize=False)

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs["scores"].cpu().float().item()


pipeline_registry: Dict[str, BaseRewardPipeline] = {
    "BaseRewardPipeline": BaseRewardPipeline
}
