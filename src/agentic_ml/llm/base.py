from abc import ABC, abstractmethod
class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, messages): pass
