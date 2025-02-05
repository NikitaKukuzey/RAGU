from abc import ABC, abstractmethod\

from environment import Registrable


class Chunker(ABC, Registrable):
    def __init__(self, config: dict=None):
        self.config = config 
    
    @abstractmethod
    def get_chunks(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_chunks(*args, **kwargs)