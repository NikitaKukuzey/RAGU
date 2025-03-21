import logging
import requests
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

import ragu.common.settings
from ragu.common.llm import BaseLLM
from ragu.common.decorator import no_throw
from ragu.common.batch_generator import BatchGenerator
from ragu.triplet.base_triplet import TripletExtractor
from ragu.utils.parse_json_output import extract_json


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    Extracts entities and relationships from text using LLM with absolute chunk indexing.
    """
    ENTITY_COLUMNS = ["entity_name", "entity_type", "entity_description", "chunk_id"]
    RELATION_COLUMNS = ["source_entity", "target_entity", "relationship_description", "chunk_id"]

    def __init__(
        self,
        entity_list_type: str,
        batch_size: int,
        validate: bool=False,
    ):
        """
        Initializes the TripletLLM extractor.

        :param validate: Flag to enable triplet validation
        :param entity_list_type: Type of entities to extract
        :param batch_size: Number of texts to process per batch
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import prompts
        from ragu.utils.default_prompts.triplet_maker_prompts import validation_prompts

        super().__init__()
        self.validate = validate
        self.batch_size = batch_size
        self.entity_list_type = entity_list_type
        self.system_prompts = prompts[self.entity_list_type]
        self.validation_system_prompts = validation_prompts[self.entity_list_type]

    def extract_entities_and_relationships(
        self,
        chunks_df: pd.DataFrame,
        client: BaseLLM=None,
        *args,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes text in batches while preserving original corpus indices.

        :param chunks_df: DataFrame containing text chunks after splitting by chunker
        :param client: LLM client instance
        :return: Tuple containing two DataFrames:
            - Entities DataFrame with columns: entity_name, entity_type, chunk_id
            - Relationships DataFrame with columns: source_entity, target_entity,
              relationship_description, chunk_id
        :raises ValueError: If required prompts are not initialized
        """

        entities, relations = [], []
        text_df = chunks_df["chunk"]
        chunks_id_df = chunks_df["chunk_id"]
        batch_generator = BatchGenerator(text_df.tolist(), batch_size=self.batch_size)

        for batch_idx, batch in tqdm(
            enumerate(batch_generator.get_batches()),
            desc="Index creation: extracting entities and relationships",
            total=len(batch_generator)
        ):
            raw_data = self._get_batched_raw_entities_and_relations(batch, client)
            parsed_batch = self._parse_llm_response(raw_data)
            self._process_parsed_batch(parsed_batch, chunks_id_df, entities, relations)

        return self._finalize_dataframes(entities, relations)

    def _get_batched_raw_entities_and_relations(self, batch: List[str], client: BaseLLM) -> List[str]:
        """
        Processes a single batch through LLM pipeline.

        :param batch: List of text chunks in current batch
        :param client: LLM client instance
        :return: Raw LLM responses for the batch
        """
        raw_data = client.generate(batch, self.system_prompts)
        return self.validate_triplets(batch, raw_data, client) if self.validate else raw_data

    def _process_parsed_batch(
        self,
        parsed_batch: List[Tuple[pd.DataFrame, pd.DataFrame]],
        chunks_id_df: pd.DataFrame,
        entities: List[pd.DataFrame],
        relations: List[pd.DataFrame]
    ) -> None:
        """
        Processes parsed batch data with absolute indexing.

        :param parsed_batch: Parsed data from current batch
        :param entities: List to accumulate entity DataFrames
        :param relations: List to accumulate relationship DataFrames
        """
        for i, (entity_df, relation_df) in enumerate(parsed_batch):
            entity_df["chunk_id"] = chunks_id_df.iloc[i]
            relation_df["chunk_id"] = chunks_id_df.iloc[i]

            entities.append(entity_df)
            relations.append(relation_df)

    def _finalize_dataframes(
        self,
        entities: List[pd.DataFrame],
        relations: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates final output DataFrames from accumulated results.

        :param entities: List of entity DataFrames
        :param relations: List of relationship DataFrames
        :return: Tuple of concatenated DataFrames
        """
        entities = pd.concat(entities, ignore_index=True) if entities else pd.DataFrame(columns=self.ENTITY_COLUMNS)
        relations = pd.concat(relations, ignore_index=True) if relations else pd.DataFrame(columns=self.RELATION_COLUMNS)

        entities.dropna(inplace=True)
        relations.dropna(inplace=True)

        # Removing "ё" is additional normalization for Russian language
        entities["entity_name"] = entities["entity_name"].str.upper().str.replace('Ё', 'Е')
        relations["source_entity"] = relations["source_entity"].str.upper().str.replace('Ё', 'Е')
        relations["target_entity"] = relations["target_entity"].str.upper().str.replace('Ё', 'Е')

        return entities, relations

    @no_throw
    def validate_triplets(
        self,
        text: List[str],
        raw_triplets: List[str],
        client: BaseLLM
    ) -> List[str]:
        """
        Validates extracted triplets using LLM.

        :param text: Original text chunks
        :param raw_triplets: Extracted triplets to validate
        :param client: LLM client instance
        :return: Validated triplet data
        """
        validation_inputs = [
            f"Text:\n{t}\n\nTriplets:\n{rt}"
            for t, rt in zip(text, raw_triplets)
        ]
        return client.generate(validation_inputs, self.validation_system_prompts)

    @no_throw
    def _parse_llm_response(self, batched_raw_data: List[str]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Parses individual LLM responses while handling errors.

        :param batched_raw_data: Raw LLM responses for a batch
        :return: List of parsed (entities, relationships) DataFrames
        """

        parsed_batch = []
        for raw in batched_raw_data:
            try:
                data = extract_json(raw)
                entities = pd.DataFrame(data["entities"])
                relations = pd.DataFrame(data["relationships"])
            except Exception as e:
                logging.error(f"Parse error: {e}\nRaw data: {raw}")
                entities = pd.DataFrame(columns=self.ENTITY_COLUMNS[:-1])
                relations = pd.DataFrame(columns=self.RELATION_COLUMNS[:-1])
            finally:
                parsed_batch.append((entities, relations))

        return parsed_batch


# TODO: add relation extraction and definition generation
@TripletExtractor.register("composed")
class ComposedTripletExtractor(TripletExtractor):
    def __init__(self, entity_list_type: str, ner_url: str):
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
        for i, text_chunk in tqdm(enumerate(texts), desc="Index creation: extracting entities and relationships", total=len(texts)):
            entities_df = self.request_ner(text_chunk)
            entities_df["chunk_id"] = i
            extracted_entities.append(entities_df)

        return pd.concat(extracted_entities, ignore_index=True) if extracted_entities \
            else pd.DataFrame(columns=["entity", "entity_type", "start", "end", "chunk_id"])

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

        data = [(text[start:end], ent_type, start, end) for start, end, ent_type in entities if ent_type in self.valid_entities]
        return pd.DataFrame(data, columns=["entity", "entity_type", "start", "end"])
