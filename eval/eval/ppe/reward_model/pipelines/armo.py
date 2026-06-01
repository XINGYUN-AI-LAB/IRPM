from utils.core import register
from reward_model.pipelines.base import BaseRewardPipeline, pipeline_registry
from typing import Dict
import torch


@register("ArmoRMPipeline", pipeline_registry)
class ArmoRMPipeline(BaseRewardPipeline):
    
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        score = model_outputs.score.cpu().float().item()
        return score
