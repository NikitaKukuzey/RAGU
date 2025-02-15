import pandas as pd
from typing import List, Any
from tqdm import tqdm

from ragu.common.settings import settings
from ragu.common.llm import BaseLLM
from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    Extracts triplets using a large language model (LLM).
    """
    
    def __init__(self, class_name: str, validate: bool) -> None:
        """
        Initializes the LLM-based triplet extractor.
        """
        super().__init__()
        self.validate = validate

    def extract_entities_and_relationships(self, elements: List[str], client: BaseLLM) -> pd.DataFrame:
        """
        Uses an LLM to extract entities and relationships from the input text and returns a DataFrame.

        :param elements: The input text to process.
        :param client: External API client for LLM interaction.
        :return: A pandas DataFrame with extracted relations.
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import tripler_system_prompts
        from ragu.utils.triplet_parser import parse_relations

        results = []

        for i, text in tqdm(enumerate(elements), desc='Index create: extract entities'):
            raw_relations = client.generate(text, tripler_system_prompts)
            extracted_relations = parse_relations(raw_relations)

            for relation in extracted_relations:
                results.append((*relation, i))

        df = pd.DataFrame(results, columns=[
            "Source entity",
            "Source entity type",
            "Relation",
            "Relation type",
            "Target entity",
            "Target entity type",
            "Chunk index"
        ])

        return df

