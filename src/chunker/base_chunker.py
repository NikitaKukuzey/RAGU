from abc import ABC, abstractmethod


class BaseChunker(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_chunks(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.get_chunks(*args, **kwargs)