import logging

import re
import pandas as pd
from typing import List, Tuple

import requests
from tqdm import tqdm

import ragu.common.settings
from ragu.common.llm import BaseLLM
from ragu.common.decorator import no_throw
from ragu.common.batch_generator import BatchGenerator
from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    A class for extracting entities and relationships (triplets) from text using a large language model (LLM).
    """
    def __init__(self, class_name: str, validate: bool, entity_list_type: str, batch_size: int):
        """
        Initializes the TripletLLM extractor.

        :param class_name: Name of the class (used for registration).
        :param validate: Whether to validate the extracted triplets (not used in this implementation).
        """
        super().__init__()
        self.validate = validate
        self.batch_size = batch_size
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

        entities, relations = [], []
        triplet_system_prompts = prompts[self.entity_list_type]
        batch_generator = BatchGenerator(text, batch_size=self.batch_size)
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Index creation: extracting entities and relationships",
                total=len(batch_generator)
        ):
            raw_data = client.generate(batch, triplet_system_prompts)

            if self.validate:
                raw_data = self.validate_triplets(batch, raw_data, client)

            parsed_data = self._parse_llm_response(raw_data)
            if parsed_data is None:
                continue

            current_chunk_entities, current_chunk_relations = parsed_data
            entities.append(current_chunk_entities)
            relations.append(current_chunk_relations)

        return pd.concat(entities, ignore_index=True), pd.concat(relations, ignore_index=True)

    @no_throw
    def validate_triplets(self, text: list[str], raw_triplets: list[str], client: BaseLLM):
        """
        Validate the triplets extracted from the text using the LLM.
        :param raw_triplets:
        :param text:
        :param client:
        :return:
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import validation_prompts

        batch_text = [f"Текст:\n{text}\n\nТриплеты:\n{raw_triplets}" for (text, raw_triplets) in zip(text, raw_triplets)]
        validation_triplet_system_prompts = validation_prompts[self.entity_list_type]
        return client.generate(batch_text, validation_triplet_system_prompts)


    @no_throw
    def _parse_llm_response(self, batched_raw_data: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses the raw LLM response to extract entities and relationships.

        :param batched_raw_data: Raw response from the LLM.
        :return: A tuple containing two DataFrames:
                 - The first DataFrame contains extracted entities with columns 'entity_name', 'entity_type', and 'chunk_id'.
                 - The second DataFrame contains extracted relationships with columns 'source_entity', 'target_entity', 'relationship_description', and 'chunk_id'.
        """

        @no_throw
        def parse(raw_data: str):
            sections = re.split(r'<\|\|>', raw_data)
            entities_section = sections[1].strip()
            relationships_section = sections[2].strip()

            entities, relationships = [], []
            entity_lines = entities_section.split('\n')
            for line in entity_lines:
                match = re.match(r'(.+?) <\|> (.+?) <\|> (.+)', line)
                if match:
                    entity_name, entity_type, entity_description = match.groups()
                    entities.append({
                        "entity_name": entity_name.strip(),
                        "entity_type": entity_type.strip(),
                        "entity_description": entity_description.strip()
                    })

            relationship_lines = relationships_section.split('\n')
            for line in relationship_lines:
                match = re.match(r'(.+?) <\|> (.+?) <\|> (.+?) <\|> (\d+)', line)
                if match:
                    source_entity, target_entity, relationship_description, relationship_strength = match.groups()
                    relationships.append({
                        "source_entity": source_entity.strip(),
                        "target_entity": target_entity.strip(),
                        "relationship_description": relationship_description.strip(),
                        "relationship_strength": int(relationship_strength.strip())
                    })

            return pd.DataFrame(entities), pd.DataFrame(relationships)

        batch_entities, batch_relations  = [], []
        for raw_data in batched_raw_data:
            parsed_data = parse(raw_data)
            if parsed_data is None:
                continue
            entities, relationships = parsed_data
            batch_relations.append(relationships)
            batch_entities.append(entities)

        return pd.concat(batch_entities), pd.concat(batch_relations)


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

        for i, text_chunk in tqdm(enumerate(texts), desc="Index creation: extracting entities and relationships"):
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
