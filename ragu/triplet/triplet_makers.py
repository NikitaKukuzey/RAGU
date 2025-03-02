import logging

import pandas as pd
from typing import List, Tuple

import requests
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
        self.validate = validate

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
        for i, text in tqdm(enumerate(text), desc='Extracting entities and relationships', total=len(text)):
            raw_data = client.generate(text, triplet_system_prompts)

            if self.validate:
                raw_data = self.validate_triplets(text, raw_data, client)

            current_chunk_entities, current_chunk_relations = parse_llm_response(raw_data)

            # Add chunk ID to track the source of each entity and relationship
            current_chunk_entities['chunk_id'] = i
            current_chunk_relations['chunk_id'] = i

            entities.append(current_chunk_entities)
            relations.append(current_chunk_relations)

        entities = pd.concat(entities)
        relations = pd.concat(relations)

        return entities, relations

    def validate_triplets(self, text: str, raw_triplets: str, client: BaseLLM):
        """
        Validate the triplets extracted from the text using the LLM.
        :param raw_triplets:
        :param text:
        :param client:
        :return:
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import validation_prompts

        prompt = "Текст:\n" + text + "\n\nТриплеты:\n" + raw_triplets
        validation_triplet_system_prompts = validation_prompts[self.entity_list_type]
        return client.generate(prompt, validation_triplet_system_prompts)


# TODO: add relation extraction and definition generation
@TripletExtractor.register("composed")
class ComposedTripletExtractor(TripletExtractor):
    def __init__(self, class_name: str, entity_list_type: str, ner_url: str):
        super().__init__()
        self.entity_list_type = entity_list_type
        self.ner_url = ner_url

        # Use english version of the dictionary because bond_005 NER returns only English entity types
        from ragu.utils.default_prompts.triplet_maker_prompts import english_entities_dict
        self.valid_entities = english_entities_dict.get(entity_list_type, set())

    def extract_entities_and_relationships(self, texts: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        Extract entities from a list of texts using a named entity recognition (NER) API.
        """
        extracted_entities = []

        for i, text_chunk in tqdm(enumerate(texts), desc="Extracting entities and relationships"):
            entities_df = self.request_ner(text_chunk)
            entities_df["chunk_id"] = i
            extracted_entities.append(entities_df)

        return pd.concat(extracted_entities, ignore_index=True) if extracted_entities \
            else pd.DataFrame(columns=["entity", "entity_type", "chunk_id"])

    def request_ner(self, text: str) -> pd.DataFrame:
        """
        Sends a request to the NER API and extracts valid entities from the response.
        """
        try:
            response = requests.post(self.ner_url, json=text)
            response.raise_for_status()
            entities = response.json().get("ners", [])
        except requests.RequestException as e:
            logging.error(f"NER request failed: {e}")
            return pd.DataFrame(columns=["entity", "entity_type"])

        data = [(text[start:end], ent_type) for start, end, ent_type in entities if ent_type in self.valid_entities]
        return pd.DataFrame(data, columns=["entity", "entity_type"])
