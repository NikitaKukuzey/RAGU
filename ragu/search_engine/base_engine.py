from abc import abstractmethod, ABC


class BaseEngine(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def build_index(self):
        """
        Build index for knowledge graph
        """
        pass

    @abstractmethod
    def search(self, query, *args, **kwargs):
        """
        Get relevant information from knowledge graph
        """
        pass

    @abstractmethod
    def query(self, query: str):
        """
        Get answer on query from knowledge graph
        """
        pass
