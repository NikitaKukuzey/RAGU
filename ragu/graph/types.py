from dataclasses import dataclass
from typing import List, Tuple, Hashable


@dataclass
class Entity:
    id: int
    entity_name: str
    entity_type: str
    description: str
    source_chunk_id: list[str]
    clusters: list


@dataclass
class Relation:
    source_entity: Entity
    target_entity: Entity
    description: str


@dataclass
class Community:
    """
    Represents a community extracted from a graph.

    Attributes:
        entities: A list of tuples where each tuple contains an entity_name ID and its description.
        relations: A list of tuples where each tuple represents an edge in the format (source_entity, target_entity, label).
    """
    level: int
    cluster_id: int
    entities: List[Tuple[Hashable, str]]
    relations: List[Tuple[Hashable, Hashable, str]]


@dataclass
class CommunitySummary:
    id: int
    summary: str

