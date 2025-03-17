import json
import logging
import pickle
from typing import List, Optional, Any

import networkx as nx
from pyvis.network import Network

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
from ragu.common.logger import log_outputs

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
            client: Optional[BaseLLM],
            chunker = None,
            triplet_extractor = None,
            reranker=None,
            generator = None,
            graph_builder = None,
    ) -> None:
        """
        Initializes the GraphRag pipeline with configured components for chunking, triplet extraction,
        reranking, and generation.

        """
        self.client = client
        self.chunker = chunker
        self.triplet = triplet_extractor
        self.reranker = reranker
        self.generator = generator
        self.graph_builder = graph_builder

        self.graph = None
        self.community_summary: Optional[dict] = None
        self.community_summary_index = None
        self.bm_25_index = None

    def from_parameters(
            self,
            chunker_parameters: Optional[ChunkerParameters] = None,
            triplet_extractor_parameters: Optional[TripletExtractorParameters] = None,
            reranker_parameters: Optional[RerankerParameters] = None,
            generator_parameters: Optional[GeneratorParameters] = None,
            graph_builder_parameters: Optional[GraphParameters] = None,
    ) -> "GraphRag":
        """
        Initializes the GraphRag pipeline with configured components for chunking, triplet extraction,
        reranking, and generation.

        :param chunker_parameters: Configuration parameters for the chunker.
        :param triplet_extractor_parameters: Configuration parameters for the triplet extractor.
        :param reranker_parameters: Configuration parameters for the reranker.
        :param generator_parameters: Configuration parameters for the generator.
        :param graph_builder_parameters: Configuration parameters for the graph builder.
        """

        self.chunker = Chunker.get(**chunker_parameters) if chunker_parameters else None
        self.triplet = TripletExtractor.get(**triplet_extractor_parameters) if triplet_extractor_parameters else None
        self.reranker = Reranker.get(**reranker_parameters) if reranker_parameters else None
        self.generator = Generator.get(**generator_parameters) if generator_parameters else None
        self.graph_builder = GraphBuilder(client=self.client, **graph_builder_parameters) if graph_builder_parameters else None

        return self

    def build(self, documents: List[str]) -> "GraphRag":
        """
        Builds a knowledge graph from a list of documents by chunking, extracting triplets,
        summarizing entities and relationships, and constructing the graph.

        :param documents: List of textual documents to process.
        :return: Instance of GraphRag with the built graph and community summaries.
        """
        if self.chunker is None or self.triplet is None or self.reranker is None or self.generator is None:
            raise ValueError("Please provide all required components for building the graph.")

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
        self.community_summary_index, self.bm_25_index = self.graph_builder.build_index(self.community_summary)

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

    def save_index(self, path: str) -> "GraphRag":
        with open(path, "wb") as f:
            pickle.dump({
                "community_summary_index": self.community_summary_index,
                "bm25_index": self.bm_25_index
            }, f)
        return self

    def load_index(self, path: str) -> "GraphRag":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.community_summary_index = data["community_summary_index"]
        self.bm_25_index = data["bm25_index"]
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
            logging.log(
                logging.ERROR,
                f"No summary available for the specified level: {which_level}. Get 0 level summary instead."
            )
            community_summary = self.community_summary.get(0)

        # Use the reranker to retrieve relevant chunks summary the community summaries
        relevant_summaries = self.reranker(
            query,
            community_summary,
            summary_index=self.community_summary_index,
            bm25=self.bm_25_index,
            embedder=self.graph_builder.embedder_model
        )
        return self.generator(query, relevant_summaries, client)

    def visualize(self, where_to_save: str = "knowledge_graph.html") -> "GraphRag":
        """
        Save a visualization of the knowledge graph in the HTML file.
        """
        if self.graph is None:
            raise RuntimeError("Graph is not built. Please build or load the graph first.")

        net = Network()
        net.from_nx(self.graph)
        net.show(where_to_save)

        return self