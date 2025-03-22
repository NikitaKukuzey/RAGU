import json
from collections import defaultdict
from dataclasses import dataclass
from typing import  Optional, Dict

import networkx as nx
import pandas as pd
from graspologic.partition import HierarchicalClusters, hierarchical_leiden
from pyvis.network import Network

from ragu.common.types import Community

@dataclass
class GraphArtifacts:
    chunks: pd.DataFrame
    entities: pd.DataFrame
    relations: pd.DataFrame

class KnowledgeGraph:
    """
    A pipeline for building, querying, and visualizing a knowledge graph using extracted triplets
    and community-based summarization.
    """

    def __init__(
            self,
            path_to_graph: Optional[str] = None,
            path_to_community_summary: Optional[str] = None
    ) -> None:
        """
        Initializes the KnowledgeGraph pipeline with configured components for chunking and triplet extraction
        """
        self.graph: nx.Graph = None
        self.community_summary: dict = None
        self.artifacts: "GraphArtifacts" = None

        if path_to_graph:
            self.load_graph(path_to_graph)
        if path_to_community_summary:
            self.load_community_summary(path_to_community_summary)

    def merge(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        if self.graph is None:
            self.graph = other.graph
        else:
            self.graph = nx.compose(self.graph, other.graph)

        return self

    def load_knowledge_graph(
            self,
            path_to_graph: str,
            path_to_community_summary: Optional[str] = None
    ) -> "KnowledgeGraph":
        """
        Loads a previously saved knowledge graph and optionally its community summary.

        :param path_to_graph: Path to the saved graph file.
        :param path_to_community_summary: Path to the saved community summary file (optional).
        """
        self.load_graph(path_to_graph)
        if path_to_community_summary:
            self.load_community_summary(path_to_community_summary)

        return self

    def load_graph(self, path: str) -> "KnowledgeGraph":
        """
        Loads a knowledge graph from a GML file.

        :param path: Path to the GML file.
        """
        self.graph = nx.read_gml(path)
        return self

    def save_graph(self, path: str) -> "KnowledgeGraph":
        """
        Saves the current knowledge graph to a GML file.

        :param path: Path where the graph will be saved.
        """
        nx.write_gml(self.graph, path)
        return self

    def save_community_summary(self, path: str) -> "KnowledgeGraph":
        """
        Saves the community summary to a json.

        :param path: Path where the summary will be saved.
        """
        if self.community_summary is None:
            raise ValueError("No community summary available to save.")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.community_summary, f, ensure_ascii=False, indent=4)

        return self

    def load_community_summary(self, path: str) -> "KnowledgeGraph":
        """
        Loads the community summary from a json.

        :param path: Path to the saved summary file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.community_summary = {int(k): v for k, v in json.load(f).items()}

        return self

    def visualize(self, where_to_save: str = "knowledge_graph.html") -> "KnowledgeGraph":
        """
        Save a visualization of the knowledge graph in the HTML file.
        """
        if self.graph is None:
            raise RuntimeError("Graph is not built. Please build or load the graph first.")

        net = Network()
        net.from_nx(self.graph)
        net.show(where_to_save)

        return self

    def detect_communities(self) -> Dict[int, Dict[int, Community]]:
        """
        Detect hierarchical communities in a graph using the Leiden algorithm.

        This function identifies communities at multiple hierarchical levels and returns
        a nested dictionary containing nodes and edges for each community at every level.

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
        community_mapping: HierarchicalClusters = hierarchical_leiden(self.graph)

        # Structure: level -> cluster_id -> (nodes, edges)
        clusters = defaultdict(lambda: defaultdict(lambda: {"nodes": set(), "edges": set()}))
        for partition in community_mapping:
            level = partition.level
            cluster_id = partition.cluster
            node = partition.node

            self.graph.nodes[node]["level"] = level
            self.graph.nodes[node]["cluster"] = cluster_id

            clusters[level][cluster_id]["nodes"].add(node)

            for neighbor in self.graph.neighbors(node):
                if neighbor in clusters[level][cluster_id]["nodes"]:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    clusters[level][cluster_id]["edges"].add(
                        (node, neighbor, edge_data.get("description", ""))
                    )

        # Convert to Community objects
        communities: Dict[int, Dict[int, Community]] = defaultdict(dict)

        for level in clusters:
            for cluster_id in clusters[level]:
                nodes = clusters[level][cluster_id]["nodes"]
                edges = clusters[level][cluster_id]["edges"]

                subgraph = self.graph.subgraph(nodes)

                entities = [
                    (node, data.get("description", ""))
                    for node, data in subgraph.nodes(data=True)
                ]

                relations = [
                    (u, v, desc)
                    for u, v, desc in edges
                ]

                communities[level][cluster_id] = Community(
                    entities=entities,
                    relations=relations
                )

        return dict(communities)