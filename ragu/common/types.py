from dataclasses import dataclass
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


@dataclass
class Relation:
    source: Node
    target: Node
    description: str

