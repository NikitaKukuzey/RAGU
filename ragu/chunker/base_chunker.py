from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ragu.common.register import Registrable
from ragu.common.types import Chunk


class Chunker(ABC, Registrable):
    """
    Abstract base class for text chunking strategies.
    Should be subclassed with specific chunking implementations.
    """
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def split(self, documents: str | List[str]) -> List[str]:
        """
        Abstract method for splitting documents into smaller chunks.
        Must be implemented in subclasses.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        pass

    def __call__(self, documents: str | List[str]) -> List[str]:
        """
        Calls the chunker on a given list of documents.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        return self.split(documents)



