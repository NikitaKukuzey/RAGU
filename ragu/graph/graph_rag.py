import json
import logging

import networkx as nx
from typing import List, Optional, Any

import pandas as pd

from ragu import (
    Chunker,
    TripletExtractor,
    Reranker,
    Generator,
)

from ragu.common.parameters import (
    ChunkerParameters,
    TripletExtractorParameters,
    RerankerParameters,
    GeneratorParameters,
    GraphParameters,
)

from ragu.graph.build import GraphBuilder
from ragu.common.llm import BaseLLM
from ragu.common.settings import log_outputs

from ragu.graph.graph_items import (
    EntitySummarizer,
    RelationSummarizer,
    get_nodes,
    get_edges,
)


class GraphRag:
    """
    A pipeline for building, querying, and visualizing a knowledge graph using extracted triplets
    and community-based summarization.
    """

    def __init__(
            self,
            client: BaseLLM,
            chunker_parameters: ChunkerParameters,
            triplet_extractor_parameters: TripletExtractorParameters,
            reranker_parameters: RerankerParameters,
            generator_parameters: GeneratorParameters,
            graph_builder_parameters: GraphParameters,
    ) -> None:
        """
        Initializes the GraphRag pipeline with configured components for chunking, triplet extraction,
        reranking, and generation.

        :param chunker_parameters: Configuration parameters for the chunker.
        :param triplet_extractor_parameters: Configuration parameters for the triplet extractor.
        :param reranker_parameters: Configuration parameters for the reranker.
        :param generator_parameters: Configuration parameters for the generator.
        """
        self.client = client

        self.chunker = Chunker.get(**chunker_parameters)
        self.triplet = TripletExtractor.get(**triplet_extractor_parameters)
        self.reranker = Reranker.get(**reranker_parameters)
        self.generator = Generator.get(**generator_parameters)
        self.graph_builder = GraphBuilder(client=client, **graph_builder_parameters)

        self.graph = None
        self.community_summary: Optional[dict] = None

    def build(self, documents: List[str]) -> "GraphRag":
        """
        Builds a knowledge graph from a list of documents by chunking, extracting triplets,
        summarizing entities and relationships, and constructing the graph.

        :param documents: List of textual documents to process.
        :return: Instance of GraphRag with the built graph and community summaries.
        """

        chunks = self.chunker(documents)
        entities, relationships = self.triplet(chunks, client=self.client)
        log_outputs(entities, "entities")
        log_outputs(relationships, "relationships")

        # TODO: move EntitySummarizer and RelationSummarizer to graph_builder
        entities = EntitySummarizer.extract_summary(
            entities,
            client=self.client,
            batch_size=self.graph_builder.batch_size
        )
        relationships = RelationSummarizer.extract_summary(
            relationships,
            client=self.client,
            batch_size=self.graph_builder.batch_size
        )
        log_outputs(entities, "summarized_entities")
        log_outputs(relationships, "summarized_relationships")

        nodes = get_nodes(entities)
        edges = get_edges(relationships, nodes)

        self.graph, self.community_summary = self.graph_builder(edges)

        return self

    def __call__(self, query: str, client: BaseLLM) -> Any:
        """
        Processes a user query by retrieving relevant information from the knowledge graph
        and generating a response.

        :param query: User query string.
        :param client: API client for response generation.
        :return: Generated response based on the query.
        """
        return self.get_response(query, client)

    def load_knowledge_graph(
            self,
            path_to_graph: str,
            path_to_community_summary: Optional[str] = None
    ) -> "GraphRag":
        """
        Loads a previously saved knowledge graph and optionally its community summary.

        :param path_to_graph: Path to the saved graph file.
        :param path_to_community_summary: Path to the saved community summary file (optional).
        """
        self.load_graph(path_to_graph)
        if path_to_community_summary:
            self.load_community_summary(path_to_community_summary)

        return self

    def load_graph(self, path: str) -> "GraphRag":
        """
        Loads a knowledge graph from a GML file.

        :param path: Path to the GML file.
        """
        self.graph = nx.read_gml(path)
        return self

    def save_graph(self, path: str) -> "GraphRag":
        """
        Saves the current knowledge graph to a GML file.

        :param path: Path where the graph will be saved.
        """
        nx.write_gml(self.graph, path)
        return self

    def save_community_summary(self, path: str) -> "GraphRag":
        """
        Saves the community summary to a json.

        :param path: Path where the summary will be saved.
        """
        if self.community_summary is None:
            raise ValueError("No community summary available to save.")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.community_summary, f, ensure_ascii=False, indent=4)

        return self

    def load_community_summary(self, path: str) -> "GraphRag":
        """
        Loads the community summary from a json.

        :param path: Path to the saved summary file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.community_summary = {int(k): v for k, v in json.load(f).items()}

        return self

    def get_response(self, query: str, client: Any, which_level: int = 0) -> Any:
        """
        Retrieves relevant information from the knowledge graph in response to a query
        and generates a response using the generator.

        :param which_level:
        :param query: User query string.
        :param client: API client for response generation.
        :return: Generated response.
        """
        if self.community_summary is None:
            raise RuntimeError("Graph is not built. Please build or load the graph first.")


        community_summary = self.community_summary.get(which_level, None)
        if community_summary is None:
            logging.log(logging.ERROR, f"No summary available for the specified level: {which_level}. Get 0 level summary instead.")
            community_summary = self.community_summary.get(0)

        # Use the reranker to retrieve relevant chunks from the community summary
        relevant_chunks = self.reranker(query, community_summary)
        return self.generator(query, relevant_chunks, client)

    def visualize(self) -> "GraphRag":
        """
        Visualizes the knowledge graph with node degree coloring using matplotlib.
        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        degrees = dict(self.graph.degree())

        min_degree = min(degrees.values())
        max_degree = max(degrees.values())
        if max_degree == min_degree:
            normalized_degrees = [0.5 for _ in degrees.values()]
        else:
            normalized_degrees = [
                (degree - min_degree) / (max_degree - min_degree)
                for degree in degrees.values()
            ]

        colors = ["#d8d8b3", "#006400"]
        custom_cmap = LinearSegmentedColormap.from_list("BeigeGreen", colors)
        colormap = custom_cmap
        node_colors = [colormap(norm_degree) for norm_degree in normalized_degrees]

        fig, ax = plt.subplots()

        pos = nx.kamada_kawai_layout(self.graph)

        nx.draw(
            self.graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            node_size=2000,
            font_size=10,
            font_weight='bold'
        )

        norm = plt.Normalize(vmin=min_degree, vmax=max_degree)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Node Degree")

        plt.title("GML Graph Visualization with Node Degree Coloring")
        plt.show()

        return self