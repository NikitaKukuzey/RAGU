import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict, Union

from ragu.common.settings import settings
from ragu.common.types import Node, Relation
from ragu.utils.triplet_parser import parse_description


def get_nodes(entities: pd.DataFrame) -> List[Node]:
    nodes: list[Node] = []

    for i, row in entities.iterrows():
        nodes.append(
            Node(
                id=int(i),
                entity=row['Entity'],
                description=row['Description']
            )
        )

    return nodes


def get_relationships(relations: pd.DataFrame, nodes: List[Node]) -> List[Relation]:
    relationships: List[Relation] = []

    # Convert to dict to speedup search
    entity_to_node: Dict[str, Node] = {node.entity: node for node in nodes}
    for _, row in relations.iterrows():
        source_node = entity_to_node.get(row['Source entity'])  # type: ignore
        target_node = entity_to_node.get(row['Target entity']) # type: ignore
        relation = row['Relation type'] # type: ignore

        if source_node and target_node:
            relationships.append(
                Relation(
                    source=source_node,
                    target=target_node,
                    description=relation
                )
            )

    return relationships


class EntityExtractor:
    @staticmethod
    def extract(raw_data: pd.DataFrame, chunks: List[str], client: Any) -> pd.DataFrame:
        """
        Extracts, merges, and summarizes entity descriptions from raw data.
        """

        __INPUT__COLUMNS__ = ['Source entity', 'Relation', 'Target entity', 'Chunk index']

        data = raw_data[['Source entity', 'Relation', 'Target entity', 'Chunk index']]
        entities_description = EntityExtractor.get_description(data, chunks, client)
        entities_description.to_csv('temp/entities_description.csv')
        return EntityExtractor.summarize(entities_description, client)

    @staticmethod
    def get_description(data: pd.DataFrame, chunks: List[str], client: Any) -> pd.DataFrame:
        """
        Extracts and combines descriptions for entities from raw data using an LLM client.
        """
        from ragu.utils.default_prompts.entites_prompts import (
            entites_info_prompt
        )

        report_data = []

        for chunk_id, group in tqdm(data.groupby('Chunk index')):
            entities = pd.concat([
                group['Source entity'],
                group['Target entity']]
            ).unique().tolist()

            text = f"""
            Список сущностей: {entities}
            Текст: {chunks[int(chunk_id)]}
            """

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

        report = pd.DataFrame(report_data, columns=['Entity', 'Description'])
        return EntityExtractor.merge_entities(report)

    @staticmethod
    def merge_entities(entities: pd.DataFrame) -> pd.DataFrame:
        """
        Groups entities by name, aggregating descriptions into a single string.
        """
        return entities.groupby("Entity")["Description"].apply("".join).reset_index()

    @staticmethod
    def summarize(data: pd.DataFrame, client: Any) -> pd.DataFrame:
        """
        Summarizes descriptions of entities using an LLM client.
        """
        from ragu.utils.default_prompts.entites_prompts import (
            entities_description_summary_prompt
        )

        summaries = []

        for _, row in tqdm(data.iterrows(), desc='Summarizing entity descriptions'):
            entity, description = row['Entity'], row['Description']

            text = f"Сущность: {entity}\nОписание: {description}"
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": entities_description_summary_prompt},
                    {"role": "user", "content": text}
                ]
            )
            summaries.append({
                'Entity': entity,
                'Description': response.choices[0].message.content
            })

        summarization_table = pd.DataFrame(summaries)
        summarization_table.to_csv('temp/summarization_table.csv', index=False)
        return summarization_table


class RelationExtractor:
    @staticmethod
    def extract(triplets: List[Dict[str, str]], client: Any) -> pd.DataFrame:
        """
        Extracts and merges relationships from triplets.
        """
        return RelationExtractor.merge_relationships(triplets)

    @staticmethod
    def merge_relationships(
            relationships: Union[pd.DataFrame, List[Dict[str, str]]]
    ) -> pd.DataFrame:
        """
        Groups relationships by source and target entities, aggregating relations into a list.
        """
        df = pd.DataFrame(relationships, columns=[
            "Source entity",
            "Source entity type",
            "Relation",
            "Relation type",
            "Target entity",
            "Target entity type",
            "Chunk index"]) \
            if not isinstance(relationships, pd.DataFrame) else relationships
        return df.groupby(["Source entity", "Target entity"]).agg({"Relation": list, "Relation type": str}).reset_index()
