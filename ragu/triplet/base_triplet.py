from abc import ABC, abstractmethod
from typing import Tuple

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
    def extract_entities_and_relationships(self, chunks_df: pd.DataFrame, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstract method for extracting entities and relationships from a given text.
        Must be implemented in subclasses.

        :param chunks_df:
        :return: A list of extracted triplets.
        """
        pass

    def __call__(self, chunks_df: pd.DataFrame, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a list of textual elements and extracts entities and relationships from each.

        :param chunks_df: List of input text elements.
        :return: A list of extracted triplets from all elements.
        """
        return self.extract_entities_and_relationships(chunks_df, *args, **kwargs)
