import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

import ragu.common.settings
from ragu.common.llm import BaseLLM
from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    A class for extracting entities and relationships (triplets) from text using a large language model (LLM).
    """
    def __init__(self, class_name: str, validate: bool, entity_list_type: str):
        """
        Initializes the TripletLLM extractor.

        :param class_name: Name of the class (used for registration).
        :param validate: Whether to validate the extracted triplets (not used in this implementation).
        """
        super().__init__()
        self.entity_list_type = entity_list_type

    def extract_entities_and_relationships(self, text: List[str], client: BaseLLM) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts entities and relationships from a list of text chunks using an LLM.

        :param text: List of text chunks to process.
        :param client: API client for interacting with the LLM.
        :return: A tuple containing two DataFrames:
                 - The first DataFrame contains extracted entities with columns 'entity_name', 'entity_type', and 'chunk_id'.
                 - The second DataFrame contains extracted relationships with columns 'source_entity', 'target_entity', 'relationship_description', and 'chunk_id'.
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import prompts
        from ragu.utils.triplet_parser import parse_llm_response

        entities = []
        relations = []
        triplet_system_prompts = prompts[self.entity_list_type]
        print(triplet_system_prompts)
        for i, text in tqdm(enumerate(text), desc='Extracting entities and relationships', total=len(text)):
            raw_data = client.generate(text, triplet_system_prompts)
            current_chunk_entities, current_chunk_relations = parse_llm_response(raw_data)

            # Add chunk ID to track the source of each entity and relationship
            current_chunk_entities['chunk_id'] = i
            current_chunk_relations['chunk_id'] = i

            entities.append(current_chunk_entities)
            relations.append(current_chunk_relations)

        entities = pd.concat(entities)
        relations = pd.concat(relations)

        return entities, relations