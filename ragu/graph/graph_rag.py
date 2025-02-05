import networkx as nx

from ragu import (
    Chunker,
    TripletExtractor, 
    Reranker, 
    Generator
)

from ragu.graph.build import GraphBuilder


class GraphRag:
    def __init__(self, config):
        self.config = config

        self.chunker = Chunker.get(**config.chunker)
        self.triplet = TripletExtractor.get(**config.triplet)
        self.reranker = Reranker.get(**config.reranker)
        self.generator = Generator.get(**config.generator)

        self.graph = nx.Graph()
        self.community_summary = None

    def build(self, documents: list[str], client):
        self.graph_builder = GraphBuilder(
            client=client, 
            config=self.config.graph
        )

        chunks = self.chunker(documents)
        triplets = self.triplet(chunks, client=client)
        self.graph, self.community_summary = self.graph_builder(triplets)
        
        return self

    def __call__(self, query):
        return self.get_responce(query)
    
    def load_knowlegde_graph(
            self, 
            path_to_graph: str, 
            path_to_community_summary: str=None
        ):
        self.load_graph(path_to_graph)
        if path_to_community_summary is not None:
            self.load_community_summary(path_to_community_summary)

    def load_graph(self, path: str):
        self.graph = nx.read_gml(path)

    def save_graph(self, path: str):
        nx.write_gml(self.graph, path)

    def save_community_summary(self, path: str):
        with open(path, "w") as f:
            f.write(self.community_summary)

    def load_community_summary(self, path: str):
        with open(path, "r") as f:
            self.community_summary = f.read()

    def get_responce(self, query):
        if self.community_summary is None:
            raise Exception("Graph is not built")
        
        relevant_chunks = self.reranker(query, self.community_summary)
        return self.generator(query, relevant_chunks, self.client)

    def visualize(self):
        pass