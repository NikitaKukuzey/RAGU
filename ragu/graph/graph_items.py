import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict

from ragu.common.batch_generator import BatchGenerator
from ragu.common.llm import BaseLLM
from ragu.common.types import Node, Relation


def get_nodes(entities_df: pd.DataFrame) -> List[Node]:
    """
    Converts a DataFrame of entities into a list of Node objects.

    :param entities_df: DataFrame containing entity information with columns 'entity_name' and 'entity_description'.
    :return: List of Node objects representing entities.
    """
    return [
        Node(
            id=int(idx),
            entity=row["entity_name"],
            description=row["entity_description"]
        )
        for idx, row in entities_df.iterrows()
    ]


def get_edges(relations_df: pd.DataFrame, nodes: List[Node]) -> List[Relation]:
    """
    Converts a DataFrame of relationships into a list of Relation objects.

    :param relations_df: DataFrame containing relationship data with columns 'source_entity', 'target_entity', and 'relationship_description'.
    :param nodes: List of Node objects used to map entities to their corresponding nodes.
    :return: List of Relation objects representing relationships between entities.
    """
    entity_to_node: Dict[str, Node] = {node.entity: node for node in nodes}
    relationships: List[Relation] = []

    for _, row in relations_df.iterrows():
        source_entity = row["source_entity"]
        target_entity = row["target_entity"]
        relation_desc = row["relationship_description"]

        source_node = entity_to_node.get(source_entity)
        target_node = entity_to_node.get(target_entity)

        if source_node and target_node:
            relationships.append(
                Relation(
                    source=source_node,
                    target=target_node,
                    description=relation_desc
                )
            )
    return relationships


class EntitySummarizer:
    """
    A class for summarizing entity descriptions by merging and processing them using an LLM client.
    """

    @staticmethod
    def extract_summary(entities_df: pd.DataFrame, client: BaseLLM, batch_size: int) -> pd.DataFrame:
        """
        Extracts and summarizes entity descriptions from raw data.

        :param batch_size:
        :param entities_df: DataFrame containing entity data with columns 'entity_name', 'entity_type', and 'entity_description'.
        :param client: LLM client used for summarization.
        :return: DataFrame with summarized entity descriptions.
        """
        merged_entities = EntitySummarizer.merge_entities(entities_df)
        return EntitySummarizer.summarize(merged_entities, client, batch_size)

    @staticmethod
    def merge_entities(entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges entity descriptions by concatenating descriptions for each unique entity.

        :param entities_df: DataFrame containing entity data with columns 'entity_name', 'entity_type', and 'entity_description'.
        :return: DataFrame with merged entity descriptions and a count of descriptions per entity.
        """
        return entities_df.groupby("entity_name").agg({
            "entity_type": "first",
            "entity_description": " ".join,
            "entity_name": "count"
        }).rename(columns={"entity_name": "description_count"}).reset_index()

    @staticmethod
    def summarize(data: pd.DataFrame, client, batch_size: int) -> pd.DataFrame:
        """
        Summarizes entity descriptions using an LLM client if an entity has multiple descriptions.

        :param data: DataFrame containing merged entity data with columns 'entity_name', 'entity_type', 'entity_description', and 'description_count'.
        :param client: LLM client used for summarization.
        :param batch_size: Number of rows to process in each batch.
        :return: DataFrame with updated entity descriptions after summarization.
        """
        from ragu.utils.default_prompts.entites_prompts import entities_description_summary_prompt

        single_desc = data[data["description_count"] == 1]
        multi_desc = data[data["description_count"] > 1].copy()

        batch_generator = BatchGenerator(multi_desc.to_dict(orient="records"), batch_size=batch_size)

        responses = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Index creation: summarizing entity descriptions",
                total=len(batch_generator)
        ):
            texts = [f"Сущность: {row["entity_name"]}\nОписание: {row["entity_description"]}" for row in batch]
            responses.extend(client.generate(texts, entities_description_summary_prompt))

        multi_desc.loc[:, "entity_description"] = responses

        return pd.concat([single_desc, multi_desc], ignore_index=True)


class RelationSummarizer:
    """
    A class for summarizing relationship descriptions by merging and processing them using an LLM client.
    """

    @staticmethod
    def extract_summary(relationships_df: pd.DataFrame, client: Any, batch_size: int) -> pd.DataFrame:
        """
        Extracts and summarizes relationship descriptions from raw data.

        :param batch_size:
        :param relationships_df: DataFrame containing relationship data with columns 'source_entity', 'target_entity', and 'relationship_description'.
        :param client: LLM client used for summarization.
        :return: DataFrame with summarized relationship descriptions.
        """
        merged_df = RelationSummarizer.merge_relationships(relationships_df)
        return RelationSummarizer.summarize(merged_df, client, batch_size)

    @staticmethod
    def merge_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
        """
        Merges relationship descriptions for each unique pair of source and target entities.

        :param relationships: DataFrame containing relationship data with columns 'source_entity', 'target_entity', and 'relationship_description'.
        :return: DataFrame with merged relationship descriptions and a count of descriptions per relationship pair.
        """
        merged_df = relationships.groupby(["source_entity", "target_entity"])
        def summarize_relations(group):
            return pd.Series({
                "relationship_description": " ".join(group["relationship_description"]),
                "description_count": len(group)
            })

        return merged_df.apply(summarize_relations).reset_index()

    @staticmethod
    def summarize(data: pd.DataFrame, client, batch_size: int) -> pd.DataFrame:
        """
        Summarizes relationship descriptions using an LLM client if a relationship has multiple descriptions.

        :param data: DataFrame containing merged relationship data with columns 'source_entity', 'target_entity', 'relationship_description', and 'description_count'.
        :param client: LLM client used for summarization.
        :param batch_size: Number of rows to process in each batch.
        :return: DataFrame with updated relationship descriptions after summarization.
        """
        from ragu.utils.default_prompts.entites_prompts import relationships_description_summary_prompt

        single_desc = data[data["description_count"] == 1]
        multi_desc = data[data["description_count"] > 1].copy()

        batch_generator = BatchGenerator(multi_desc.to_dict(orient="records"), batch_size=batch_size)

        responses = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Index creation: summarizing relationship descriptions",
                total=len(batch_generator)):
            texts = [
                f"""
                Source entity: {row["source_entity"]}, Target entity: {row["target_entity"]}
                Description: {row["relationship_description"]}
                """.strip() for row in batch
            ]
            responses.extend(client.generate(texts, relationships_description_summary_prompt))

        multi_desc.loc[:, "relationship_description"] = responses
        return pd.concat([single_desc, multi_desc], ignore_index=True)
