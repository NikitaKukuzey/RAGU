import networkx as nx
from tqdm import tqdm
from typing import List, Tuple, Any, Hashable

from ragu.common.types import Relation, Community
from ragu.common.settings import settings
from ragu.common.llm import BaseLLM
from ragu.utils.default_prompts.community_summary_prompt import prompt


def detect_communities(graph: nx.Graph) -> List[Community]:
    """
    Detect communities in the given graph by extracting its weakly connected components.

    Each component is converted into a Community instance containing its nodes and edges.

    :param graph: A NetworkX graph.
    :return: A list of Community instances.
    """
    communities: List[Community] = []

    for component_nodes in nx.weakly_connected_components(graph):
        component_subgraph = graph.subgraph(component_nodes)
        entities = [
            (node, data.get("description", ""))
            for node, data in component_subgraph.nodes(data=True)
        ]
        relations = [
            (u, v, data.get("label", ""))
            for u, v, data in component_subgraph.edges(data=True)
        ]
        communities.append(Community(entities=entities, relations=relations))

    return communities


def get_community_summaries(communities: List[Community], client: BaseLLM) -> List[str]:
    """
    Generate summaries for each community using a language model.

    The function builds a textual representation for each community (with its entities and relations)
    and passes it to the language model API client.

    :param communities: List of Community instances.
    :param client: API client for interacting with the language model.
    :return: List of summaries for each community.
    """

    def compose_community_string(
            entities: List[Tuple[Hashable, str]],
            relations: List[Tuple[Hashable, Hashable, str]]
    ) -> str:
        vertices_str = ",\n".join(
            f"Сущность: {entity}, описание: {description}" for entity, description in entities
        )
        relations_str = "\n".join(
            f"{source} -> {label} -> {target}" for source, target, label in relations
        )
        return f"{vertices_str}\n\nОтношения:\n{relations_str}"

    summaries = []
    for community in tqdm(communities, desc='Index create: community summary'):
        community_text = compose_community_string(community.entities, community.relations)

        summary = client.generate(community_text, prompt).strip()
        summaries.append(summary)

    return summaries


class GraphBuilder:
    """
    Graph processing pipeline that builds a graph, extracts communities,
    and generates summaries for each community.
    """

    def __init__(self, client: Any) -> None:
        """
        Initialize the GraphBuilder with the required API client and configuration.

        :param client: API client for community summarization.
        """
        self.client = client

    def __call__(self, relations: List[Relation]) -> Tuple[nx.DiGraph, List[str]]:
        """
        Execute the graph processing pipeline.

        The method builds a directed graph from the provided relations, detects communities,
        and generates summaries for each community.

        :param relations: List of Relation objects.
        :return: A tuple containing the built directed graph and a list of community summaries.
        """
        graph = self.build_graph(relations)
        communities = detect_communities(graph)
        community_summaries = get_community_summaries(communities, self.client)
        return graph, community_summaries

    def build_graph(self, relations: List[Relation]) -> nx.DiGraph:
        """
        Build a directed graph from the provided relations.

        Each relation defines source and target nodes along with edge labels.

        :param relations: List of Relation objects.
        :return: A directed NetworkX graph.
        """
        graph = nx.DiGraph()
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
                label=relation.description
            )

        return graph
