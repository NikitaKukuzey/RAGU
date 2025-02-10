from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ragu.common.register import Registrable
from ragu.common.types import Chunk


class Chunker(ABC, Registrable):
    """
    Abstract base class for text chunking strategies.
    Should be subclassed with specific chunking implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the chunker with an optional configuration.

        :param config: A dictionary containing configuration parameters.
        """
        self.config = config
    
    @abstractmethod
    def get_chunks(self, documents: List[str]) -> List[Chunk]:
        """
        Abstract method for splitting documents into smaller chunks.
        Must be implemented in subclasses.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        pass

    def __call__(self, documents: List[str]) -> List[Chunk]:
        """
        Calls the chunker on a given list of documents.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        return self.get_chunks(documents)



