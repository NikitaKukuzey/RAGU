from dataclasses import dataclass
from typing import List, Tuple, Any, Hashable
from typing import NewType


CommunityId = NewType("CommunityId", int)
Chunk = NewType("Chunk", str)


@dataclass
class CommunitySummary:
    id: CommunityId
    summary: str


@dataclass
class Node:
    id: int
    entity: str
    description: str
    source_chunk_id: list[str]
    clusters: list


@dataclass
class Relation:
    source: Node
    target: Node
    description: str


@dataclass
class Community:
    """
    Represents a community extracted from a graph.

    Attributes:
        entities: A list of tuples where each tuple contains an entity ID and its description.
        relations: A list of tuples where each tuple represents an edge in the format (source, target, label).
    """
    level: int
    cluster_id: int
    entities: List[Tuple[Hashable, str]]
    relations: List[Tuple[Hashable, Hashable, str]]

