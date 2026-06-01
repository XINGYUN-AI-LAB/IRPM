from utils.core import register
from reward_model.models.base import model_registry
from transformers import PreTrainedModel
from torch import nn
import torch

@register("NaiveVerbosityModel", model_registry)
class NaiveVerbosityModel(PreTrainedModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.PAD_ID = None
        self.config = kwargs["config"]
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        assert self.PAD_ID is not None, "Please set the pad id before calling forward."

        bs = int(input_ids.shape[0])
        scores = []
        for i in range(bs):
            c_inds = (input_ids[i] != self.PAD_ID).nonzero()
            length = len(c_inds)
            scores.append(torch.Tensor([length]).float())
        scores = torch.stack(scores)
        return {"scores": scores}