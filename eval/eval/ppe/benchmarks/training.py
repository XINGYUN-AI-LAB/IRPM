from utils.core import register
from benchmarks.base import BaseBenchmark, benchmark_registry


@register("train_best_of_k", benchmark_registry)
class TrainPreferenceBenchmark(BaseBenchmark):
    """For collecting reward model preference for the end-to-end RLHF experiment."""

    def __init__(self, bias=0, iterator=True, split="train") -> None:
        self.responses_per_question = 16
        self.benchmark_path = "./benchmarks/best_of_k_training.json"
        super().__init__(bias=bias, iterator=iterator, split=split)
