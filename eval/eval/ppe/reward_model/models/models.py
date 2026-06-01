# import torch
# from torch import nn
# from transformers import (
#     PreTrainedModel,
#     PreTrainedTokenizerBase,
#     AutoTokenizer,
#     LlamaPreTrainedModel,
#     LlamaModel,
#     AutoConfig,
#     AutoModelForCausalLM,
#     Qwen2Model,
#     Qwen2PreTrainedModel
# )
# from utils import register
# from registry import model_registry

# @register("NaiveVerbosityModel", model_registry)
# class NaiveVerbosityModel(PreTrainedModel):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.PAD_ID = None
#         self.config = kwargs["config"]
#         self.dummy = nn.Linear(1, 1)

#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         mc_token_ids=None,
#         labels=None,
#         return_dict=False,
#         output_attentions=False,
#         output_hidden_states=False,
#     ):
#         assert self.PAD_ID is not None, "Please set the pad id before calling forward."

#         bs = int(input_ids.shape[0])
#         scores = []
#         for i in range(bs):
#             c_inds = (input_ids[i] != self.PAD_ID).nonzero()
#             length = len(c_inds)
#             scores.append(torch.Tensor([length]).float())
#         scores = torch.stack(scores)
#         return {"scores": scores}


# @register("Nexusflow/Starling-RM-34B", model_registry)
# class StarlingForSequenceClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = LlamaModel(config)
#         self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
#         self.PAD_ID = 0
#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def get_device(self):
#         return self.transformer.device

#     def forward(
#           self,
#           input_ids=None,
#           past_key_values=None,
#           attention_mask=None,
#           position_ids=None,
#       ):
#           transformer_outputs = self.transformer(
#               input_ids,
#               attention_mask=attention_mask,
#               position_ids=position_ids,
#               output_hidden_states=True,
#           )
#           hidden_states = transformer_outputs.hidden_states[-1]
#           scores = []
#           rewards = self.v_head(hidden_states).squeeze(-1)
#           bs = int(input_ids.shape[0])
#           for i in range(bs):
#               c_inds = (input_ids[i] == self.PAD_ID).nonzero()
#               c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
#               scores.append(rewards[i, c_ind - 1])
#           scores = torch.stack(scores)
#           return {"scores": scores}

# @register("berkeley-nest/Starling-RM-7B-alpha", model_registry)
# class StarlingAlphaForSequenceRegression(nn.Module):
#     def __init__(self, model_path):
#         super().__init__()
#         model = AutoModelForCausalLM.from_pretrained(model_path)
#         self.config = model.config
#         self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
#         self.model = model
#         self.transformer = model.model
#         self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.tokenizer.pad_token = self.tokenizer.unk_token
#         self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

#         self.device = self.get_device()

#     def get_device(self):
#         return self.model.device
    
#     def can_generate(self):
#         return False

#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         position_ids=None,
#     ):
#         """
#         input_ids, attention_mask: torch.Size([bs, seq_len])
#         return: scores: List[bs]
#         """
#         bs = input_ids.shape[0]
#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#         )
#         hidden_states = transformer_outputs[0]
#         scores = []
#         rewards = self.v_head(hidden_states).squeeze(-1)
#         for i in range(bs):
#             c_inds = (input_ids[i] == self.PAD_ID).nonzero()
#             c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
#             scores.append(rewards[i, c_ind - 1])
#         return {"scores": torch.vstack(scores)}
    
# @register("Nexusflow/Athene-RM-70B", model_registry)
# @register("Nexusflow/Athene-RM-8B", model_registry)
# @register("Nexusflow/Llama-3.1-8B-Instruct-RM-Test", model_registry)
# @register("Nexusflow/Llama-3.1-70B-RM-09172024", model_registry)
# class AtheneForSequenceClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaModel(config)
#         self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
#         self.CLS_ID = 128003
#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def get_device(self):
#         return self.model.device

#     def forward(
#           self,
#           input_ids=None,
#           past_key_values=None,
#           attention_mask=None,
#           position_ids=None,
#       ):
#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_hidden_states=True,
#         )
#         hidden_states = transformer_outputs.hidden_states[-1]
#         scores = []
#         rewards = self.v_head(hidden_states).squeeze(-1)

#         bs = int(input_ids.shape[0])

#         for i in range(bs):
#             c_inds = (input_ids[i] == self.CLS_ID).nonzero()
#             c_ind = c_inds[-1].item()
#             scores.append(rewards[i, c_ind])
#         scores = torch.stack(scores)
#         return {"scores": scores}

# @register("Nexusflow/Qwen2.5-7B-RM-09192024", model_registry)
# @register("Nexusflow/Qwen2.5-72B-RM-09242024", model_registry)
# class QwenForSequenceClassification(Qwen2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = Qwen2Model(config)
#         self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
#         self.CLS_ID = 151665
#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def get_device(self):
#         return self.model.device

#     def forward(
#           self,
#           input_ids=None,
#           past_key_values=None,
#           attention_mask=None,
#           position_ids=None,
#       ):
#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_hidden_states=True,
#         )
#         hidden_states = transformer_outputs.hidden_states[-1]
#         scores = []
#         rewards = self.v_head(hidden_states).squeeze(-1)

#         bs = int(input_ids.shape[0])

#         for i in range(bs):
#             c_inds = (input_ids[i] == self.CLS_ID).nonzero()
#             c_ind = c_inds[-1].item()
#             scores.append(rewards[i, c_ind])
#         scores = torch.stack(scores)
#         return {"scores": scores}