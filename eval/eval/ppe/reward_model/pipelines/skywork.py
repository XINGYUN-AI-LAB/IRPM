from utils.core import register
from reward_model.pipelines.base import BaseRewardPipeline, pipeline_registry


@register("SkyworkPipeline", pipeline_registry)
class SkyworkPipeline(BaseRewardPipeline):
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        output = model_outputs["logits"][0][0]

        return output.cpu().float().item()
