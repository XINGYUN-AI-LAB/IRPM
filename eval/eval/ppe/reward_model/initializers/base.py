from abc import ABC, abstractmethod
from typing import Dict


class BaseRewardModelHandler(ABC):
    def __init__(self, model_name=None) -> None:
        self.model_name = model_name
        self.tokenizer = self.init_tokenizer()
        self.model = self.init_model()
        self.pipeline = self.init_pipeline()

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_tokenizer(self):
        pass

    @abstractmethod
    def init_pipeline(self):
        pass

    def get_artifacts(self):
        return self.tokenizer, self.model, self.pipeline


init_registry: Dict[str, BaseRewardModelHandler] = {}
