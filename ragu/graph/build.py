from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Tuple, Any, Hashable, Dict, Set

import networkx as nx
from graspologic.partition import (
    HierarchicalClusters,
    hierarchical_leiden,
)

from ragu.common.types import Relation, Community
from ragu.common.llm import BaseLLM
from ragu.common.batch_generator import BatchGenerator
from ragu.utils.parse_json_output import extract_json, combine_report_text


def detect_communities(graph: nx.Graph) -> Dict[int, Dict[int, Community]]:
    """
    Detect hierarchical communities in a graph using the Leiden algorithm.

    This function identifies communities at multiple hierarchical levels and returns
    a nested dictionary containing nodes and edges for each community at every level.

    :param graph: A NetworkX graph to analyze.
    :return: A nested dictionary where:
             - The outer key is the hierarchy level (int).
             - The inner key is the cluster ID (int).
             - The value is a Community object containing nodes and edges for the cluster.
             Example:
             {
                 level_1: {
                     cluster_id_1: Community(...),
                     cluster_id_2: Community(...)
                 },
                 level_2: {
                     cluster_id_1: Community(...)
                 }
             }
    """
    community_mapping: HierarchicalClusters = hierarchical_leiden(graph)

    # Structure: level -> cluster_id -> (nodes, edges)
    clusters = defaultdict(lambda: defaultdict(lambda: {"nodes": set(), "edges": set()}))
    for partition in community_mapping:
        level = partition.level
        cluster_id = partition.cluster
        node = partition.node

        clusters[level][cluster_id]["nodes"].add(node)

        for neighbor in graph.neighbors(node):
            if neighbor in clusters[level][cluster_id]["nodes"]:
                edge_data = graph.get_edge_data(node, neighbor)
                clusters[level][cluster_id]["edges"].add(
                    (node, neighbor, edge_data.get("description", ""))
                )

    # Convert to Community objects
    result: Dict[int, Dict[int, Community]] = defaultdict(dict)

    for level in clusters:
        for cluster_id in clusters[level]:
            nodes = clusters[level][cluster_id]["nodes"]
            edges = clusters[level][cluster_id]["edges"]

            subgraph = graph.subgraph(nodes)

            entities = [
                (node, data.get("description", ""))
                for node, data in subgraph.nodes(data=True)
            ]

            relations = [
                (u, v, desc)
                for u, v, desc in edges
            ]

            result[level][cluster_id] = Community(
                entities=entities,
                relations=relations
            )

    return dict(result)


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

        :param entities: A list of tuples containing entity names and descriptions.
        :param relations: A list of tuples containing source, target, and relation descriptions.
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

    summaries = []
    batch_generator = BatchGenerator(communities, batch_size=batch_size)
    for batch in tqdm(batch_generator.get_batches(), desc="Index creation: community summary", total=len(batch_generator)):
        community_text = [compose_community_string(community.entities, community.relations) for community in batch]
        summary = client.generate(community_text, generate_community_report)
        summary = [(extract_json(s)) for s in summary]
        summaries.extend(summary)
    return summaries


class GraphBuilder:
    """
    A pipeline for building a graph, detecting communities, and generating summaries.

    This class provides functionality to:
    1. Build a graph from a list of relations.
    2. Detect hierarchical communities using the Leiden algorithm.
    3. Generate summaries for each community using a language model.
    """

    def __init__(
            self,
            client: Any,
            batch_size: int=16,
            which_level: int = -1,
            embedder_model_name: str=None,
            enable_index: bool = True,
            **kwargs
    ) -> None:
        """
        Initialize the GraphBuilder.

        :param client: An API client for community summarization.
        :param batch_size: Batch size for LLM processing.
        :param which_level: The hierarchy level to analyze (-1 for all levels).
        :param embedder_model_name: Name of the embedding model.
        :param enable_index: Whether to build an index for retrieval.
        """
        self.client = client
        self.batch_size = batch_size
        self.which_level = which_level
        self.enable_index = enable_index

        if embedder_model_name:
            self.embedder_model = SentenceTransformer(
                embedder_model_name,
                **kwargs
            )

    def __call__(self, relations: List[Relation]):
        """
        Execute the graph processing pipeline.

        This method builds a graph, detects communities, and generates summaries.

        :param relations: A list of Relation objects representing graph edges.
        :return: A tuple containing:
                 - The built directed graph (nx.DiGraph).
                 - A list of community summaries (List[str]).
        """
        graph = self.build_graph(relations)
        communities = detect_communities(graph)

        if self.which_level == -1:
            levels = list(communities.keys())
        else:
            levels = [self.which_level]

        community_summaries = {}
        for level in levels:
            list_of_communities_at_level = list(communities.get(level).values())
            community_summaries_at_level = get_community_summaries(
                list_of_communities_at_level,
                self.client,
                self.batch_size
            )
            community_summaries[level] = community_summaries_at_level

        return graph, community_summaries

    def build_index(self, summaries: dict):
        if self.embedder_model and self.enable_index:
            docs = [combine_report_text(report) for report in summaries.get(self.which_level, [])]
            return self.embedder_model.encode(docs), BM25Okapi([doc.split() for doc in docs])

    def build_graph(self, relations: List[Relation]) -> nx.Graph:
        """
        Build a directed graph from a list of relations.

        Each relation defines a source node, a target node, and an edge description.

        :param relations: A list of Relation objects.
        :return: A directed NetworkX graph.
        """
        graph = nx.Graph()
        for relation in relations:
            source_entity = relation.source.entity
            target_entity = relation.target.entity

            graph.add_node(
                source_entity,
                entity_name=source_entity,
                description=relation.source.description
            )

            graph.add_node(
                target_entity,
                entity_name=target_entity,
                description=relation.target.description
            )

            graph.add_edge(
                source_entity,
                target_entity,
                description=relation.description
            )

        return graph