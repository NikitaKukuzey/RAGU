from abc import ABC, abstractmethod
from typing import  List, Tuple

import pandas as pd

from ragu.common.register import Registrable


class TripletExtractor(ABC, Registrable):
    """
    Abstract base class for extracting entities and relationships from text.
    This class should be subclassed with an implementation of the extraction method.
    """
    
    def __init__(self) -> None:
        """
        Initializes the triplet extractor with an optional configuration.
        """

    @abstractmethod
    def extract_entities_and_relationships(self, text: List[str], **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstract method for extracting entities and relationships from a given text.
        Must be implemented in subclasses.

        :param text: The input text to process.
        :return: A list of extracted triplets.
        """
        pass

    def __call__(self, text: List[str], **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a list of textual elements and extracts entities and relationships from each.

        :param text: List of input text elements.
        :return: A list of extracted triplets from all elements.
        """
        return self.extract_entities_and_relationships(text, **kwargs)
