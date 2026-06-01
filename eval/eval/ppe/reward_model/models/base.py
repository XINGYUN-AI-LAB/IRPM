from typing import Dict
from transformers import PreTrainedModel
from torch import nn

model_registry: Dict[str, PreTrainedModel | nn.Module] = {}