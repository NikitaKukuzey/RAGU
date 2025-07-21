from ragu.utils.io_utils import read_text_from_files
from ragu.common.llm import RemoteLLM
from ragu.graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph

from ragu.search_engine.local_search import LocalSearchEngine
from ragu.common.embedder import STEmbedder
from ragu.chunker.chunkers import SmartSemanticChunker, JsonPassChunker
from ragu.triplet.triplet_makers import TripletLLM, JsonTripletLLM
from ragu.common.index import Index

import pandas as pd
import numpy as np
import json


# You can load your creditals from .env. Look into ragu/common/setting.py
LLM_MODEL_NAME = "..."
LLM_BASE_URL = "..."
LLM_API_KEY = "..."
client = RemoteLLM(LLM_MODEL_NAME, LLM_BASE_URL, LLM_API_KEY)

# Getting documents from folders with .txt files
text = read_text_from_files('/path/to/data/folder')

# Initialize a chunker
chunker = SmartSemanticChunker(
    reranker_name="/path/to/reranker_model", #BAAI/bge-reranker-v2-m3 - рекомендовано использовать
    max_chunk_length=1024
)

#Если уже есть файлы в json с ответами модели
#chunker = JsonPassChunker()

# Initialize a triplet extractor
artifact_extractor = JsonTripletLLM( #наша реализация
    validate=False,
    # Also you can set your own entity types as a list (and others arguments)
    # entity_list_type=[your entity types],
    # batch_size=16
)

#Если уже есть файлы в json с ответами модели
#artifact_extractor = JsonPassTripletLLM(validate=False)

# Initialize a graph builder pipeline
graph_builder = KnowledgeGraphBuilder(
    client,
    triplet_extractor=artifact_extractor,
    chunker=chunker
)

# Run building
knowledge_graph = graph_builder.build(text)

# Save results
knowledge_graph.save_graph("graph.gml").save_community_summary("summary.json")