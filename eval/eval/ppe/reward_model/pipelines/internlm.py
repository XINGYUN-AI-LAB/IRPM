from utils.core import register
from reward_model.pipelines.base import BaseRewardPipeline, pipeline_registry
from typing import Dict
import torch


@register("InternLM2Pipeline", pipeline_registry)
class InternLM2Pipeline(BaseRewardPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs[0], dict):
            inputs = [inputs]
        conversation_strs = [
            self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            for conversation in inputs
        ]
        batch_input_ids = []
        attention_masks = []

        for conversation_str in conversation_strs:
            input_ids = self.tokenizer.encode(
                conversation_str,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=4095,
                truncation=True,
            )
            # add reward score token at the end of the input_ids if it is not already there
            if input_ids[0, -1] != self.model.reward_token_id:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor([[self.model.reward_token_id]], dtype=torch.long),
                    ],
                    dim=1,
                )
            input_ids = input_ids.squeeze(0)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool)
            batch_input_ids.append(input_ids)
            attention_masks.append(attention_mask)

        r_pad_batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        r_pad_attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=False
        )

        return {
            "input_ids": r_pad_batch_input_ids,
            "attention_mask": r_pad_attention_masks,
        }

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        scores = model_outputs[0].cpu().float().item()
        return scores
