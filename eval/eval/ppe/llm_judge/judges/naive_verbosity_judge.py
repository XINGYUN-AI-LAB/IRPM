from typing import Dict, List, Literal, Tuple
from utils.core import register
from llm_judge.judges.base import BaseJudge, judge_registry


@register("naive-verbosity-judge", judge_registry)
class NaiveVerbosityJudge(BaseJudge):

    def __init__(self, prompt_path: str = None):
        return

    def _format(self, user_prompt: str, responses: List[str]) -> List[Dict[str, str]]:
        return

    def _parse_judgment(self, judgment: str) -> Literal[0] | Literal[1]:
        return

    def judge(
        self,
        user_prompt: str,
        responses: List[str],
        temperature: float,
        api_type: str,
        api_dict: Dict,
        model_name: str,
        max_tokens=8192,
    ) -> Tuple[Literal[0, 1], str]:

        pick_longer = int(len(responses[0]) > len(responses[1]))

        return pick_longer, "I like this one cuz its more verbose! I'm so naive!"
