from typing import Type

from ragu.common.embedder import BaseEmbedder
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.storage.base_storage import BaseKVStorage, BaseVectorStorage
from ragu.storage.json_storage import JsonKVStorage
from ragu.storage.nano_db import NanoVectorDBStorage
from ragu.utils.hash import compute_mdhash_id


class Index:
    """
    Index class that manages embedding storage and indexing for a knowledge graph.
    """
    def __init__(
            self,
            embedder: BaseEmbedder,
            kv_storage_type: Type[BaseKVStorage] = JsonKVStorage,
            vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage,
            chunks_kv_storage_kwargs: dict = None,
            summary_kv_storage_kwargs: dict = None,
            vdb_storage_kwargs: dict = None,
    ):
        """
        Initializes the Index with storage configurations and an embedding model.

        :param embedder: The embedding model to use.
        :param kv_storage_type: The key-value storage type, defaults to JsonKVStorage.
        :param vdb_storage_type: The vector database storage type, defaults to NanoVectorDBStorage.
        :param chunks_kv_storage_kwargs: Configuration for chunk storage, defaults to None.
        :param summary_kv_storage_kwargs: Configuration for summary storage, defaults to None.
        :param vdb_storage_kwargs: Configuration for vector database storage, defaults to None.
        """

        DEFAULT_FILENAMES = {
            "community_summary_db": "community_summary_kv.json",
            "chunks_db": "chunks_kv.json",
            "entity_db": "entity_vdb.json",
        }
        self.embedder = embedder

        self.summary_kv_storage_kwargs = summary_kv_storage_kwargs or {}
        self.chunks_kv_storage_kwargs = chunks_kv_storage_kwargs or {}
        self.vdb_storage_kwargs = vdb_storage_kwargs or {}

        if self.summary_kv_storage_kwargs.get("filename") is None:
            self.summary_kv_storage_kwargs["filename"] = DEFAULT_FILENAMES["community_summary_db"]
        if self.chunks_kv_storage_kwargs.get("filename") is None:
            self.chunks_kv_storage_kwargs["filename"] = DEFAULT_FILENAMES["chunks_db"]
        if self.vdb_storage_kwargs.get("filename") is None:
            self.vdb_storage_kwargs["filename"] = DEFAULT_FILENAMES["entity_db"]

        self.entity_vector_db = vdb_storage_type(embedder=embedder, **self.vdb_storage_kwargs)
        self.community_summary_kv_storage = kv_storage_type(**self.summary_kv_storage_kwargs)
        self.chunks_kv_storage = kv_storage_type(**self.chunks_kv_storage_kwargs)

    async def make_index(self, knowledge_graph: KnowledgeGraph):
        """
        Creates an index for the given knowledge graph by storing entities, chunks, and summaries.

        :param knowledge_graph: The knowledge graph containing data to index.
        """
        if self.entity_vector_db is not None:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["entity_description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in knowledge_graph.artifacts.entities.to_dict("records")
            }
            await self.entity_vector_db.upsert(data_for_vdb)

        if self.chunks_kv_storage is not None:
            data_for_kv = {
                dp["chunk_id"]: dp["chunk"]
                for dp in knowledge_graph.artifacts.chunks.to_dict("records")
            }
            await self.chunks_kv_storage.upsert(data_for_kv)

        if self.community_summary_kv_storage is not None:
            data_for_kv = knowledge_graph.community_summary
            await self.community_summary_kv_storage.upsert(data_for_kv)

        await self.community_summary_kv_storage.index_done_callback()
        await self.chunks_kv_storage.index_done_callback()
        await self.entity_vector_db.index_done_callback()
