import asyncio
import string
from itertools import combinations
from typing import List

import pandas as pd

from ragu.search_engine.base_engine import BaseEngine
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.triplet.base_triplet import TripletExtractor
from ragu.common.llm import BaseLLM
from ragu.search_engine.types import SearchResult
from ragu.utils.hash import compute_mdhash_id


class BFSEngine(BaseEngine):
    """
    A search engine that performs breadth-first search (BFS) over a knowledge graph.
    """
    def __init__(
            self,
            knowledge_graph: KnowledgeGraph,
            artifacts_extractor: TripletExtractor,
            client: BaseLLM=None,
            *args,
            **kwargs
    ):
        """
        Initializes the BFS-based search engine.

        :param knowledge_graph: The knowledge graph used for search operations.
        :param artifacts_extractor: Extractor for retrieving entities and relationships.
        :param client: LLM client for generating responses, defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.knowledge_graph = knowledge_graph
        self.artifacts_extractor = artifacts_extractor
        self.client = client

    def build_index(self) -> "BFSEngine":
        """
        Builds an index for the search engine. Not implemented.
        """
        return self

    def search(self, query: str, depth: int=2, *args, **kwargs):
        """
        Performs a BFS search on the knowledge graph given a query.

        :param query: The search query.
        :param depth: The maximum depth of the BFS traversal, defaults to 2.
        :return: A tuple containing entities and relations found in the search.
        """
        query_entities = self._get_query_entities(query)
        entities, relations = self.get_neighbors_bfs(
            graph=self.knowledge_graph.graph,
            query_entities=query_entities,
            max_depth=depth
        )
        return SearchResult(entities=entities, relations=relations, summaries=[], chunks=[])

    def query(self, query: str, depth: int=2):
        """
        Processes a query using BFS search and generates a response using the LLM client.

        :param depth:
        :param query: The input query.
        :return: The generated response from the LLM.
        """
        context: SearchResult = self.search(query, depth)

        entity_context, relationship_context = context.entities, context.relations

        entities_list = "\n".join([f"{entity[0]}, {entity[1]}" for entity in entity_context])
        relationship_list = "\n".join([f"{entity[0]}, {entity[1]}, {entity[2]}" for entity in relationship_context])
        context = f"Сущности:\n{entities_list}\n\nОтношения:\n{relationship_list}\n\n"

        sys_prompt = "Ты - полезный ассистент. Ответь на запрос по предоставленному контексту."
        user_prompt = f"Запрос:\n{query}\n\nКонтекст:\n{context}"
        return self.client.generate(user_prompt, sys_prompt)[0]

    @staticmethod
    def _get_all_possible_entities(entities: List[str]):
        """
        Generates all possible combinations of entities.

        :param entities: List of entities.
        :return: A list of all possible entity_name combinations.
        """

        unique_phrases = set()
        for s in entities:
            cleaned = s.translate(str.maketrans('', '', string.punctuation))
            words = cleaned.split()
            unique_phrases.add(cleaned)

            for i in range(1, len(words) + 1):
                for combo in combinations(words, i):
                    unique_phrases.add(" ".join(combo))

        return list(unique_phrases)

    def _get_query_entities(self, query: str):
        """
        Extracts entities from the query using the artifacts extractor.

        :param query: The input query.
        :type query: str
        :return: A list of extracted entity_name names.
        """
        query = pd.DataFrame(
            {
                "chunk": [query],
                "chunk_id": compute_mdhash_id(query)
            }
        )
        entities, _ = self.artifacts_extractor.extract_entities_and_relationships(query, client=self.client)
        entities = entities.drop_duplicates('entity_name')
        return self._get_all_possible_entities(entities["entity_name"].tolist())

    @staticmethod
    def get_neighbors_bfs(graph, query_entities: str | List[str], max_depth):
        """
        Performs a breadth-first search (BFS) on a graph to retrieve neighboring entities and relationships.

        :param graph: The knowledge graph to traverse.
        :param query_entities: List of starting entities for BFS.
        :param max_depth: Maximum depth for BFS traversal.
        :return: A tuple containing nodes (entities) and edges (relationships) found in the search.
        """

        if isinstance(query_entities, str):
            query_entities = [query_entities]

        nodes_list = []
        edges_list = []
        visited = set()

        queue = [(entity, 0) for entity in query_entities if entity in graph]
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue

            visited.add(current)
            nodes_list.append((current, graph.nodes[current].get("description", "")))

            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    edge_data = graph.get_edge_data(current, neighbor)
                    edges_list.append((current, neighbor, edge_data.get("description", "")))
                    queue.append((neighbor, depth + 1))

        return nodes_list, edges_list
