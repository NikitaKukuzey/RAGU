# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

import asyncio
import logging

from ragu.common.index import Index
from ragu.common.llm import BaseLLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.common.embedder import BaseEmbedder
from ragu.search_engine.search_functional import (
    _find_most_related_text_unit_from_entities,
    _find_most_related_edges_from_entities,
)

logging.basicConfig(level=logging.INFO)


class LocalSearchEngine(BaseEngine):
    """
    Perform local search on a knowledge graph using a given query.

    Reference: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """
    def __init__(
        self,
        client: BaseLLM,
        knowledge_graph: KnowledgeGraph,
        embedder: BaseEmbedder,
        index: Index,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.community_reports = None
        self.llm_response_cache = None
        self.text_chunks = None
        self.full_docs = None
        self.graph = knowledge_graph
        self.embedder = embedder
        self.index = index
        self.client = client

    async def build_index(self):
        pass

    async def search(self, query: str, top_k: int = 20, *args, **kwargs) -> str:
        """
        Find the most related entities/chunks/relations to the given query.

        :param query: The query to search for.
        :param top_k: The number of relevant artifacts used to build local context.
        :return: A text report containing the most related entities/chunks/relations.
        """
        entities = await self.index.entity_vector_db.query(query, top_k=top_k)
        nodes_data = [
            {**n, "entity_name": k["entity_name"]}
            for k in entities
            if (n := self.graph.graph.nodes.get(k["entity_name"])) is not None
        ]

        chunks = await _find_most_related_text_unit_from_entities(
            nodes_data, self.index.chunks_kv_storage, self.graph
        )
        relations = await _find_most_related_edges_from_entities(nodes_data, self.graph)

        entity_section = "Сущность, тип сущности, описание сущности\n" + "\n".join(
            f"{n['entity_name']}, {n.get('entity_type', 'UNKNOWN')}, {n.get('description', 'UNKNOWN')}"
            for n in nodes_data
        )

        relations_section = "Сущность, целевая сущность, описание отношения, ранг\n" + "\n".join(
            f"{e['source']}, {e['target']}, {e.get('description', 'UNKNOWN')}, {e.get('rank', 'UNKNOWN')}"
            for e in relations
        )

        chunks_section = "\n\n".join(chunks)
        return (
            f"**Сущности**:\n {entity_section}\n"
            f"**Отношения**:\n{relations_section}\n"
            f"**Тексты**:\n{chunks_section}\n"
        )

    def query(self, query: str) -> str:
        """
        Perform RAG on knowledge graph using the local context.
        :param query: user query
        :return: RAG response
        """
        from ragu.utils.default_prompts.search_engine_query_prompts import system_prompt, query_prompts

        context = asyncio.run(self.search(query))
        return self.client.generate(
            query_prompts.format(query=query, context=context),
            system_prompt
        )
