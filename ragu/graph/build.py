import networkx as nx
from tqdm import tqdm
from cdlib import algorithms
from ragu.common.types import Relation
from typing import List, Tuple, Callable, Any

from ragu.common.settings import settings

# TODO: Add new component content composing and insert it in GraphBuilder
def detect_communities(graph: nx.Graph) -> List[str]:
    """
    Detects communities in the graph and represents them as strings.

    Each community is a set of connected nodes, and the function extracts
    edge information to create a textual representation of each community.

    :param graph: NetworkX graph.
    :return: List of community representations as strings.
    """
    # communities_content = []
    # for component in nx.connected_components(graph):
    #     subgraph = graph.subgraph(component)
    #     # Entities:
    #     #entities = [f'Сущность: {u}, описание: {descr}' for u, descr in subgraph.nodes(data=True)]
    #     # print(subgraph)
    #     component_content = [f"{u} {attrs['label']} {v}" for u, v, attrs in subgraph.edges(data=True)]
    #     communities_content.append(". ".join(component_content))
    # return communities_content
    pass


# TODO: Insert it in GraphBuilder
def get_community_summary(texts: List[str], client: Any, config: Any) -> List[str]:
    """
    Generates summaries for detected communities using a language model.

    :param texts: List of community descriptions.
    :param client: API client for interacting with the language model.
    :param config: Configuration object containing model parameters.
    :return: List of summaries for each community.
    """
    from ragu.utils.default_prompts.community_summary_prompt import prompt

    summaries = []
    for text in tqdm(texts, desc='Index create: community summary'):
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        summaries.append(response.choices[0].message.content.strip())
    return summaries


class GraphBuilder:
    """
    Graph processing pipeline that builds a graph, detects communities,
    and generates summaries for each community.
    """

    def __init__(
            self,
            client: Any,
            config: Any,
    ) -> None:
        """
        Initializes the GraphBuilder with the necessary components.

        :param client: API client for community summarization.
        :param config: Configuration object containing model parameters.
        :param graph_builder_func: Function to construct a graph from triplets.
        :param get_community_summary_func: Function to summarize detected communities.
        :param detect_communities_func: Function to detect communities in a graph.
        """
        self.client = client
        self.config = config
        self.graph_builder_func = graph_builder_func
        self.get_community_summary_func = get_community_summary_func
        self.detect_communities_func = detect_communities_func

    def __call__(
            self,
            relations,
            *args,
            **kwargs
    ) -> Tuple[nx.Graph, List[str]]:
        """
        Executes the graph processing pipeline.

        :param triplets: List of (entity1, relation, entity2) tuples.
        :return: Tuple containing the built graph and a list of community summaries.
        """
        graph = self.build(relations)
        # communities = self.detect_communities_func(graph)
        # communities_summaries = self.get_community_summary_func(
        #     communities,
        #     self.client,
        #     self.config
        # )
        return graph, communities_summaries

    def build(self, relations: list[Relation]):
        graph = nx.Graph()

        for relation in relations:
            graph.add_node(
                relation.source.entity,
                entity_name=relation.source.entity,
                description=relation.source.description
            )
            graph.add_node(
                relation.target.entity,
                entity_name=relation.target.entity,
                description=relation.target.description
            )
            graph.add_edge(
                relation.source.entity,
                relation.target.entity,
                label=relation.description
            )

        return graph
