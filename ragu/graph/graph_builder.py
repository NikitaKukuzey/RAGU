import asyncio
import os
from typing import List, Tuple, Any, Hashable, Dict, Type
from dataclasses import asdict

import pandas as pd
import networkx as nx
from tqdm import tqdm

from ragu.chunker.base_chunker import Chunker
from ragu.storage.base_storage import BaseKVStorage
from ragu.storage.json_storage import JsonKVStorage
from ragu.triplet.base_triplet import TripletExtractor
from ragu.common.llm import BaseLLM
from ragu.graph.types import Relation, Entity, Community
from ragu.utils.parse_json_output import extract_json
from ragu.utils.default_prompts.artifact_summarization_prompts import artifacts_summarization_prompt
from ragu.common.batch_generator import BatchGenerator
from ragu.graph.knowledge_graph import KnowledgeGraph, GraphArtifacts
from ragu.common.logger import log_outputs
from ragu.common.global_parameters import storage_run_dir, DEFAULT_FILENAMES
from ragu.common.logger import logging


class GraphConstructor:
    """
    Builds a NetworkX graph from given relations.
    """

    @staticmethod
    def build_graph(relations: List[Relation]) -> nx.Graph:
        """
        Build an undirected graph from a list of relations.

        :param relations: List of Relation objects.
        :return: A NetworkX graph.
        """
        graph = nx.Graph()
        for relation in relations:
            source = relation.source_entity.entity_name
            target = relation.target_entity.entity_name

            target_node_data = asdict(relation.target_entity)
            source_node_data = asdict(relation.source_entity)
            target_node_data.pop("entity_name")
            source_node_data.pop("entity_name")

            graph.add_node(source, **source_node_data)
            graph.add_node(target, **target_node_data)
            graph.add_edge(
                source,
                target,
                description=relation.description,
                weight=relation.relation_strength
            )
        return graph


class CommunitySummarizer:
    """
    Generates summaries for communities using an LLM client.
    """

    @staticmethod
    def get_community_summaries(communities: List[Community], client: BaseLLM, batch_size: int) -> List[dict]:
        """
        Generate summaries for each community using a language model (LLM).

        This function composes a textual representation of each community (including its entities
        and relations) and uses the LLM client to generate a summary.

        :param communities: A list of Community objects to summarize.
        :param client: An API client for interacting with the LLM.
        :param batch_size: The batch size for processing summaries.
        :return: A list of summaries, one for each community.
        """

        def compose_community_string(
                entities: List[Tuple[Hashable, str]],
                relations: List[Tuple[Hashable, Hashable, str]]
        ) -> str:
            """
            Compose a textual representation of a community.

            :param entities: A list of tuples containing entity_name names and descriptions.
            :param relations: A list of tuples containing source_entity, target_entity, and relation descriptions.
            :return: A formatted string representing the community.
            """
            vertices_str = ",\n".join(
                f"Сущность: {entity}, описание: {description}" for entity, description in entities
            )
            relations_str = "\n".join(
                f"{source} -> {target}, описание отношения: {description}" for source, target, description in relations
            )
            return f"{vertices_str}\n\nОтношения:\n{relations_str}"

        from ragu.utils.default_prompts.community_summary_prompt import generate_community_report

        summaries: list[dict] = []
        batch_generator = BatchGenerator(communities, batch_size=batch_size)
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Community summary",
                total=len(batch_generator)
        ):
            community_text = [
                compose_community_string(community.entities, community.relations)
                for community in batch
            ]

            generated_community_summaries = client.generate(
                community_text,
                generate_community_report,
            )

            for community, raw_generated_community_summary in zip(batch, generated_community_summaries):
                parsed = extract_json(raw_generated_community_summary)
                if parsed is None:
                    logging.warning(f"Bad JSON: {raw_generated_community_summary}")

                summaries.append(
                    {
                        "cluster_id": community.cluster_id,
                        "level": community.level,
                        "community_report": parsed,
                        "entities": community.entities,
                        "relations": community.relations
                    }
                )

        if len(summaries) != len(communities):
            logging.warning(
                f"Some community were missed. Number of communities in graph is {len(communities)}, but number of summaries is {len(summaries)}"
            )

        return summaries


class EntitySummarizer:
    """
    A class for summarizing entity descriptions by merging and processing them using an LLM client.
    """

    @staticmethod
    def extract_summary(entities_df: pd.DataFrame, client: BaseLLM, batch_size: int,
                        summarize_with_llm: bool = False) -> pd.DataFrame:
        """
        Extracts and summarizes entity_name descriptions from raw data.

        :param entities_df: DataFrame containing entity_name data with columns 'entity_name', 'entity_type', and 'description'.
        :param client: LLM client used for summarization.
        :return: DataFrame with summarized entity_name descriptions.
        :param summarize_with_llm: Whether to summarize the entity_name descriptions with the LLM client.
        :param batch_size: Number of rows to process in each batch.
        """

        merged_df = EntitySummarizer.merge_entities(entities_df)
        return EntitySummarizer.summarize(merged_df, client, batch_size) if summarize_with_llm else merged_df

    @staticmethod
    def merge_entities(entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges entity_name descriptions by concatenating descriptions for each unique entity_name.

        :param entities_df: DataFrame containing entity_name data with columns 'entity_name', 'entity_type', and 'description'.
        :return: DataFrame with merged entity_name descriptions and a count of descriptions per entity_name.
        """
        return entities_df.groupby("entity_name").agg({
            "entity_type": "first",
            "description": " ".join,
            "entity_name": "count",
            "chunk_id": lambda x: list(set(x))  # Get only unique chunks id
        }).rename(columns={"entity_name": "description_count"}).reset_index()

    @staticmethod
    def summarize(data: pd.DataFrame, client, batch_size: int) -> pd.DataFrame:
        """
        Summarizes entity_name descriptions using an LLM client if an entity_name has multiple descriptions.

        :param data: DataFrame containing merged entity_name data with columns 'entity_name', 'entity_type', 'description', 'description_count' and 'chunk id'.
        :param client: LLM client used for summarization.
        :param batch_size: Number of rows to process in each batch.
        :return: DataFrame with updated entity_name descriptions after summarization.
        """

        single_desc = data[data["description_count"] == 1]
        multi_desc = data[data["description_count"] > 1].copy()

        batch_generator = BatchGenerator(multi_desc.to_dict(orient="records"), batch_size=batch_size)

        responses = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Summarizing entity descriptions",
                total=len(batch_generator)
        ):
            texts = [f"Сущность: {row["entity_name"]}\nОписание: {row["description"]}" for row in batch]
            responses.extend(client.generate(texts, artifacts_summarization_prompt["summarize_entity_descriptions"]))

        responses = [
            parsed["description"] if (parsed := extract_json(resp)) else None
            for resp in responses
        ]
        if len(responses) != len(multi_desc):
            logging.warning(
                f"Failed to summarize some descriptions. "
                f"Number of summarized descriptions is {len(responses)}, but number of descriptions to summarize was {len(multi_desc)}"
                "Return descriptions as it is (without LLM summarization)"
            )
            return data

        multi_desc.loc[:, "description"] = responses
        multi_desc.dropna(inplace=True)

        return pd.concat([single_desc, multi_desc], ignore_index=True)


class RelationSummarizer:
    """
    A class for summarizing relationship descriptions by merging and processing them using an LLM client.
    """

    @staticmethod
    def extract_summary(relationships_df: pd.DataFrame, client: Any, batch_size: int,
                        summarize_with_llm: bool = False) -> pd.DataFrame:
        """
        Extracts and summarizes relationship descriptions from raw data.

        :param relationships_df: DataFrame containing relationship data with columns 'source_entity', 'target_entity', 'relationship_description/strength' and 'chunk_id'.
        :param client: LLM client used for summarization.
        :return: DataFrame with summarized relationship descriptions.
        :param summarize_with_llm: Whether to summarize the relation descriptions with the LLM client.
        :param batch_size: Number of rows to process in each batch.
        """
        merged_df = RelationSummarizer.merge_relationships(relationships_df)
        return RelationSummarizer.summarize(merged_df, client, batch_size) if summarize_with_llm else merged_df

    @staticmethod
    def merge_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
        """
        Merges relationship descriptions for each unique pair of source_entity and target_entity entities.

        :param relationships: DataFrame containing relationship data with columns
            'source_entity', 'target_entity', 'relationship_description' and 'chunk_id'.
        :return: DataFrame with merged relationship descriptions and a count of descriptions per relationship pair.
        """
        return relationships.groupby(["source_entity", "target_entity"]).agg({
            "relationship_description": " ".join,
            "relationship_strength": "mean",
            "chunk_id": lambda x: list(set(x))  # Get only unique chunks id
        }).assign(description_count=lambda df: df["chunk_id"].apply(len)).reset_index()

    @staticmethod
    def summarize(data: pd.DataFrame, client, batch_size: int) -> pd.DataFrame:
        """
        Summarizes relationship descriptions using an LLM client if a relationship has multiple descriptions.

        :param data: DataFrame containing relationship data with columns
            'source_entity', 'target_entity', 'relationship_description' and 'chunk_id'.
        :param client: LLM client used for summarization.
        :param batch_size: Number of rows to process in each batch.
        :return: DataFrame with updated relationship descriptions after summarization.
        """

        single_desc = data[data["description_count"] == 1]
        multi_desc = data[data["description_count"] > 1].copy()

        batch_generator = BatchGenerator(multi_desc.to_dict(orient="records"), batch_size=batch_size)

        responses = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Summarizing relationship descriptions",
                total=len(batch_generator)):
            texts = [
                f"""
                Сущность: {row["source_entity"]}, Целевая сущность: {row["target_entity"]}
                Описание: {row["relationship_description"]}
                """.strip() for row in batch
            ]
            responses.extend(client.generate(texts, artifacts_summarization_prompt["summarize_relation_descriptions"]))

        responses = [
            parsed["description"] if (parsed := extract_json(resp)) else None
            for resp in responses
        ]

        if len(responses) != len(multi_desc):
            logging.warning(
                f"Failed to summarize some descriptions. "
                f"Number of summarized descriptions is {len(responses)}, but number of descriptions to summarize was {len(multi_desc)}"
                "Return descriptions as it is (without LLM summarization)"
            )
            return data

        multi_desc.loc[:, "relationship_description"] = responses
        multi_desc.dropna(inplace=True)

        return pd.concat([single_desc, multi_desc], ignore_index=True)


class KnowledgeGraphBuilder:
    """
    A pipeline for building a knowledge graph.

    :param client: BaseLLM client instance for external API calls.
    :param batch_size: Number of elements to process in each batch.
    :param chunker: Component responsible for splitting documents into chunks.
    :param triplet_extractor: Component responsible for extracting triplets from chunks.
    :param summarize_entities: Flag to enable summarization of entity_name descriptions.
    :param summarize_relations: Flag to enable summarization of relationship descriptions.
    """

    def __init__(
            self,
            client: BaseLLM,
            chunker: Chunker,
            triplet_extractor: TripletExtractor,
            take_strongest_components: bool = False,
            remove_isolated_nodes: bool = False,
            max_cluster_size: int = 512,
            summarize_entities: bool = True,
            summarize_relations: bool = True,
            batch_size: int = 16,
            save_intermediate_results: bool = True,
            kv_storage_type: Type[BaseKVStorage] = JsonKVStorage,
            chunk_kv_storage_params: Dict[str, Any] = None,
            community_kv_storage_params: Dict[str, Any] = None,
            storage_folder: str = storage_run_dir,
    ):
        # Initialize the pipeline
        self.client = client
        self.batch_size = batch_size
        self.chunker = chunker
        self.triplet_extractor = triplet_extractor

        # Set graph parameters
        self.summarize_relations = summarize_relations
        self.summarize_entities = summarize_entities
        self.max_cluster_size = max_cluster_size
        self.remove_isolated_nodes = remove_isolated_nodes
        self.take_strongest_components = take_strongest_components

        # Set options
        self.save_intermediate_results = save_intermediate_results
        self.chunk_kv_storage_params = chunk_kv_storage_params if chunk_kv_storage_params else {}
        self.community_kv_storage_params = community_kv_storage_params if community_kv_storage_params else {}

        # Set key-value storage
        self.community_kv_storage_params["filename"] = os.path.join(storage_folder, DEFAULT_FILENAMES[
            "community_summary_kv_storage_name"])
        self.chunk_kv_storage_params["filename"] = os.path.join(storage_folder,
                                                                DEFAULT_FILENAMES["chunks_kv_storage_name"])

        self.chunks_kv_storage = kv_storage_type(**self.chunk_kv_storage_params)
        self.communities_kv_storage = kv_storage_type(**self.community_kv_storage_params)

    def from_parameters(self):
        ...

    def _get_nodes(self, entities_df: pd.DataFrame) -> List[Entity]:
        """
        Converts a DataFrame of entities into a list of Entity objects.

        :param entities_df: DataFrame containing entity_name information with columns 'entity_name' and 'description'.
        :return: List of Entity objects representing entities.
        """
        return [
            Entity(
                id=int(idx),
                entity_name=row["entity_name"],
                entity_type=row["entity_type"],
                description=row["description"],
                source_chunk_id=row["chunk_id"],
                clusters=[],
            )
            for idx, row in entities_df.iterrows()
        ]

    def _get_edges(self, relations_df: pd.DataFrame, nodes: List[Entity]) -> List[Relation]:
        """
        Converts a DataFrame of relationships into a list of Relation objects.

        :param relations_df: DataFrame containing relationship data with columns 'source_entity', 'target_entity', and 'relationship_description/strength'.
        :param nodes: List of Entity objects used to map entities to their corresponding nodes.
        :return: List of Relation objects representing relationships between entities.
        """
        entity_to_node: Dict[str, Entity] = {node.entity_name: node for node in nodes}
        relationships: List[Relation] = []

        for _, row in relations_df.iterrows():
            source_entity = row["source_entity"]
            target_entity = row["target_entity"]
            description = row["relationship_description"]
            strength=row["relationship_strength"]

            source_node = entity_to_node.get(source_entity)
            target_node = entity_to_node.get(target_entity)

            if source_node and target_node:
                relationships.append(
                    Relation(
                        source_entity=source_node,
                        target_entity=target_node,
                        description=description,
                        relation_strength=strength
                    )
                )
        return relationships

    def build(self, documents: List[str]) -> KnowledgeGraph:
        """
        Builds a knowledge graph from a list of documents by chunking, extracting triplets,
        summarizing entities and relationships, and constructing the graph.

        :param documents: List of textual documents to process.
        :return: Instance of KnowledgeGraph with the built graph and community summaries.
        """
        if self.chunker is None or self.triplet_extractor is None:
            raise ValueError("Please provide all required components for building the graph.")

        # Step 1: Split documents into chunks
        chunks = self.chunker(documents)
        logging.info("Text splitted into chunks")
        # Step 2: Extract entities, relationships, and their descriptions
        logging.info("Start extracting ER with LLM")
        entities, relationships = self.triplet_extractor(chunks, client=self.client)
        logging.info("Start summarizing graph")
        # Step 3: Summarize entities' and relationships' descriptions
        summarized_entities = EntitySummarizer.extract_summary(
            entities,
            client=self.client,
            batch_size=self.batch_size,
            summarize_with_llm=self.summarize_entities
        )
        logging.info("Entities summarized")
        summarized_relationships = RelationSummarizer.extract_summary(
            relationships,
            client=self.client,
            batch_size=self.batch_size,
            summarize_with_llm=self.summarize_relations
        )
        logging.info("Relationships summarized")
        # Step 4: Construct the graph
        nodes = self._get_nodes(summarized_entities)
        edges = self._get_edges(summarized_relationships, nodes)
        graph = GraphConstructor.build_graph(edges)

        knowledge_graph = KnowledgeGraph()
        knowledge_graph.graph = graph

        knowledge_graph.save_graph(
            os.path.join(storage_run_dir, DEFAULT_FILENAMES["graph_name"])  # type: ignore
        )

        if self.remove_isolated_nodes:
            knowledge_graph.remove_isolated_nodes()

        if self.take_strongest_components:
            knowledge_graph.take_stable_largest_connected_component()

        # Step 5: Detect communities in the graph
        communities = knowledge_graph.detect_communities()

        # Step 6: Get community summaries
        summary = CommunitySummarizer.get_community_summaries(
            communities=communities,
            client=self.client,
            batch_size=self.batch_size
        )

        knowledge_graph.community_summary = summary

        knowledge_graph.save_community_summary(
            os.path.join(storage_run_dir, DEFAULT_FILENAMES["summary_name"])  # type: ignore
        )

        knowledge_graph.artifacts = GraphArtifacts(
            entities=entities,
            relations=relationships,
            chunks=chunks
        )

        if self.save_intermediate_results:
            log_outputs(chunks, "chunks")
            log_outputs(entities, "entities")
            log_outputs(relationships, "relationships")
            log_outputs(summarized_entities, "summarized_entities")
            log_outputs(summarized_relationships, "summarized_relationships")

        asyncio.run(self.save_artifacts(chunks, summary))

        return knowledge_graph

    async def save_artifacts(
            self,
            chunks: pd.DataFrame,
            community_summary: list
    ) -> None:
        """
        Saves the artifacts (chunks, entities, and relationships) of the constructed knowledge graph.

        :param community_summary:
        :param chunks: DataFrame containing chunk information.
        """
        if self.chunks_kv_storage is not None:
            data_for_kv = {
                dp["chunk_id"]: dp["chunk"]
                for dp in chunks.to_dict("records")
            }
            await self.chunks_kv_storage.upsert(data_for_kv)

        if self.communities_kv_storage is not None:
            data_for_kv = {
                summary["cluster_id"]: summary
                for summary in community_summary
            }

            await self.communities_kv_storage.upsert(data_for_kv)

        await self.chunks_kv_storage.index_done_callback()
        await self.communities_kv_storage.index_done_callback()
