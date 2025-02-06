from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ragu.common import Registrable


class Reranker(ABC, Registrable):
    """
    Abstract base class for reranking documents based on relevance to a query.
    Should be subclassed with an implementation of the reranking method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the reranker with an optional configuration.

        :param config: A dictionary containing configuration parameters.
        """
        self.config = config

    @abstractmethod
    def get_relevant_chunks(self, query: str, documents: List[str]) -> List[str]:
        """
        Abstract method for retrieving relevant document chunks for a given query.
        Must be implemented in subclasses.

        :param query: The input query.
        :param documents: List of candidate documents.
        :return: A ranked list of relevant documents.
        """
        pass

    def __call__(self, query: str, documents: List[str]) -> List[str]:
        """
        Calls the reranker on a given query and list of documents.

        :param query: The input query.
        :param documents: List of candidate documents.
        :return: A ranked list of relevant documents.
        """
        return self.get_relevant_chunks(query, documents)