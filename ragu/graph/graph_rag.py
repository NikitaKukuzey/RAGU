import networkx as nx
from typing import List, Optional, Any

from ragu import (
    Chunker, 
    TripletExtractor, 
    Reranker, 
    Generator
)

from ragu.graph.build import GraphBuilder

from ragu.graph.graph_items import (
    EntityExtractor,
    RelationExtractor,
    get_nodes,
    get_relationships
)



class GraphRag:
    """
    A pipeline for building and querying a knowledge graph using extracted triplets
    and community-based summarization.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initializes the GraphRag pipeline components based on the provided configuration.

        :param config: Configuration object containing parameters for chunking,
                       triplet extraction, reranking, and generation.
        """
        self.config = config
        self.chunker = Chunker.get(**config.chunker)
        self.triplet = TripletExtractor.get(**config.triplet)
        self.reranker = Reranker.get(**config.reranker)
        self.generator = Generator.get(**config.generator)
        
        self.graph = nx.Graph()
        self.community_summary: Optional[str] = None

    def build(self, documents: List[str], client: Any) -> "GraphRag":
        """
        Builds the knowledge graph from a list of documents.

        :param documents: List of textual documents.
        :param client: API client for processing and summarization.
        :return: Instance of GraphRag with a built graph and community summaries.
        """
        graph_builder = GraphBuilder(
            client=client, 
            config=self.config.graph
        )
        chunks = self.chunker(documents)

        triplets = self.triplet(chunks, client=client)

        entities = EntityExtractor.extract(triplets, chunks, client=client)
        relationships = RelationExtractor.extract(triplets, client)

        nodes = get_nodes(entities)
        edges = get_relationships(relationships, nodes)

        self.graph = graph_builder.build(edges)

        return self

    def __call__(self, query: str, client: Any) -> Any:
        """
        Handles queries by retrieving relevant information from the knowledge graph.

        :param query: User query string.
        :param client: API client for response generation.
        :return: Generated response based on the query.
        """
        return self.get_response(query, client)

    def load_knowledge_graph(
            self, 
            path_to_graph: str, 
            path_to_community_summary: Optional[str] = None
        ) -> None:
        """
        Loads a previously saved knowledge graph and optionally its community summary.

        :param path_to_graph: Path to the saved graph file.
        :param path_to_community_summary: Path to the saved community summary file (optional).
        """
        self.load_graph(path_to_graph)
        if path_to_community_summary:
            self.load_community_summary(path_to_community_summary)

    def load_graph(self, path: str) -> None:
        """
        Loads a knowledge graph from a GML file.

        :param path: Path to the GML file.
        """
        self.graph = nx.read_gml(path)

    def save_graph(self, path: str) -> None:
        """
        Saves the current knowledge graph to a GML file.

        :param path: Path where the graph will be saved.
        """
        nx.write_gml(self.graph, path)

    def save_community_summary(self, path: str) -> None:
        """
        Saves the community summary to a text file.

        :param path: Path where the summary will be saved.
        """
        if self.community_summary is None:
            raise ValueError("No community summary available to save.")
        
        with open(path, "w") as f:
            f.write(self.community_summary)

    def load_community_summary(self, path: str) -> None:
        """
        Loads the community summary from a text file.

        :param path: Path to the saved summary file.
        """
        with open(path, "r") as f:
            self.community_summary = f.read()

    def get_response(self, query: str, client: Any) -> Any:
        """
        Retrieves relevant information from the knowledge graph in response to a query.

        :param query: User query string.
        :param client: API client for response generation.
        :return: Generated response.
        """
        if self.community_summary is None:
            raise RuntimeError("Graph is not built. Please build or load the graph first.")
        
        relevant_chunks = self.reranker(query, self.community_summary)
        return self.generator(query, relevant_chunks, client)

    def visualize(self) -> None:
        """
        Visualizes the knowledge graph with node degree coloring.
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
