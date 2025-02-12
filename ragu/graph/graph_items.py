import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict, Union

from ragu.common.settings import settings
from ragu.common.types import Node, Relation
from ragu.utils.triplet_parser import parse_description


def get_nodes(entities_df: pd.DataFrame) -> List[Node]:
    """
    Converts a DataFrame containing entity information into a list of Node objects.

    :param entities_df: DataFrame with columns 'Entity' and 'Description'.
    :return: List of Node objects.
    """
    return [
        Node(
            id=int(idx),
            entity=row['Entity'],
            description=row['Description']
        )
        for idx, row in entities_df.iterrows()
    ]


def get_relationships(relations_df: pd.DataFrame, nodes: List[Node]) -> List[Relation]:
    """
    Converts a DataFrame containing relationship data into a list of Relation objects.

    :param relations_df: DataFrame with columns 'Source entity', 'Target entity', 'Relation', and 'Relation type'.
    :param nodes: List of Node objects used to find corresponding entities.
    :return: List of Relation objects.
    """
    relations_df["Relation type"] = relations_df["Relation type"].astype(str)
    entity_to_node: Dict[str, Node] = {node.entity: node for node in nodes}
    relationships: List[Relation] = []

    for _, row in relations_df.iterrows():
        source_entity = row['Source entity']
        target_entity = row['Target entity']
        relation_desc = row['Relation']

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


class EntityExtractor:
    """
    Extracts, merges, and summarizes entity descriptions using an LLM client.
    """

    @staticmethod
    def extract(raw_data: pd.DataFrame, chunks: List[str], client: Any) -> pd.DataFrame:
        """
        Extracts and summarizes entity descriptions from raw data.

        :param raw_data: DataFrame with columns 'Source entity', 'Relation', 'Target entity', and 'Chunk index'.
        :param chunks: List of text fragments corresponding to raw data sections.
        :param client: API client for processing and summarization.
        :return: DataFrame containing entities and their summarized descriptions.
        """
        data = raw_data[['Source entity', 'Relation', 'Target entity', 'Chunk index']]
        entities_description = EntityExtractor.get_description(data, chunks, client)
        return EntityExtractor.summarize(entities_description, client)

    @staticmethod
    def get_description(data: pd.DataFrame, chunks: List[str], client: Any) -> pd.DataFrame:
        """
        Extracts and merges entity descriptions from data using an LLM client.

        :param data: DataFrame with 'Source entity', 'Relation', 'Target entity', and 'Chunk index' columns.
        :param chunks: List of text fragments corresponding to data sections.
        :param client: API client for entity description extraction.
        :return: DataFrame with merged entity descriptions.
        """
        from ragu.utils.default_prompts.entites_prompts import entites_info_prompt

        report_data: List[Dict[str, str]] = []

        for chunk_id, group in tqdm(data.groupby('Chunk index'), desc="Extracting entity descriptions"):
            unique_entities = pd.concat([group['Source entity'], group['Target entity']]).unique().tolist()
            text = f"Entity list: {unique_entities}\nText: {chunks[int(chunk_id)]}"

            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": entites_info_prompt},
                    {"role": "user", "content": text}
                ]
            )

            parsed_response = parse_description(response.choices[0].message.content)
            if parsed_response:
                report_data.extend(parsed_response)

        report_df = pd.DataFrame(report_data, columns=['Entity', 'Description'])
        return EntityExtractor.merge_entities(report_df)

    @staticmethod
    def merge_entities(entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges entity descriptions by concatenating descriptions for each entity.

        :param entities_df: DataFrame with columns ['Entity', 'Description'].
        :return: DataFrame with merged entity descriptions.
        """
        return entities_df.groupby("Entity", as_index=False)["Description"].agg("".join)

    @staticmethod
    def summarize(data: pd.DataFrame, client: Any) -> pd.DataFrame:
        """
        Summarizes entity descriptions using an LLM client.

        :param data: DataFrame with columns ['Entity', 'Description'].
        :param client: API client for summarization.
        :return: DataFrame with summarized entity descriptions.
        """
        from ragu.utils.default_prompts.entites_prompts import entities_description_summary_prompt
        summaries: List[Dict[str, str]] = []

        for _, row in tqdm(data.iterrows(), desc='Summarizing entity descriptions', total=len(data)):
            entity = row['Entity']
            description = row['Description']
            text = f"Entity: {entity}\nDescription: {description}"

            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": entities_description_summary_prompt},
                    {"role": "user", "content": text}
                ]
            )
            summaries.append({'Entity': entity, 'Description': response.choices[0].message.content.strip()})

        summarization_table = pd.DataFrame(summaries)
        summarization_table.to_csv('temp/summarization_table.csv', index=False)
        return summarization_table


class RelationExtractor:
    """
    Extracts and merges relationships from triplet data.
    """

    @staticmethod
    def extract(triplets: List[Dict[str, str]], client: Any) -> pd.DataFrame:
        """
        Extracts and merges relationships from a list of triplets.

        :param triplets: List of dictionaries representing relationship triplets.
        :param client: API client (not used currently, reserved for future improvements).
        :return: DataFrame containing relationship information.
        """
        return RelationExtractor.merge_relationships(triplets)

    @staticmethod
    def merge_relationships(relationships: Union[pd.DataFrame, List[Dict[str, str]]]) -> pd.DataFrame:
        """
        Merges relationship data into a DataFrame.

        :param relationships: Relationship data as a DataFrame or list of dictionaries.
        :return: DataFrame containing merged relationships.
        """

        # merged_df = df.groupby(["Source entity", "Target entity"]).agg({
        #     "Relation": list,
        #     "Relation type": "first"
        # }).reset_index()
        # return merged_df

        return pd.DataFrame(relationships) if not isinstance(relationships, pd.DataFrame) else relationships
