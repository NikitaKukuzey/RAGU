from ragu.utils.io_utils import read_text_from_files, read_text_from_chegeka
from ragu.common.llm import RemoteLLM, LocalLLM, VLLMClient
from ragu.graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph

from ragu.search_engine.local_search import LocalSearchEngine
from ragu.common.embedder import STEmbedder
from ragu.chunker.chunkers import SimpleChunker, JsonPassChunker
from ragu.triplet.triplet_makers import TripletLLM, JsonTripletLLM
from ragu.common.index import Index
from ragu.common.logger import logging

import pandas as pd
import numpy as np
import json


# You can load your creditals from .env. Look into ragu/common/setting.py
#LLM_MODEL_NAME = "..."
#LLM_BASE_URL = "..."
#LLM_API_KEY = "..."

PATH_TO_CHEGEKA = "./chegeka"

logging.info("Initializing VLLM started")
client = VLLMClient("path/to/our/model") #todo прописать путь до модели
logging.info("Initializing VLLM finished")
# Getting documents from folders with .txt files
logging.info("Parsing chegeka started")
text = read_text_from_chegeka(PATH_TO_CHEGEKA + '/' + 'documents.json')
logging.info("Parsing chegeka finished")
# Initialize a chunker
logging.info("Initializing Chunker started")
chunker = SimpleChunker(
    max_chunk_length=1024,
    overlap=0
)
logging.info("Initializing Chunker finished")
#Если уже есть файлы в json с ответами модели
#chunker = JsonPassChunker()
logging.info("Initializing JsonTripletLLM started")
# Initialize a triplet extractor
artifact_extractor = JsonTripletLLM( #наша реализация
    validate=False,
    # Also you can set your own entity types as a list (and others arguments)
    # entity_list_type=[your entity types],
    # batch_size=16
)
logging.info("Initializing JsonTripletLLM finished")
#Если уже есть файлы в json с ответами модели
#artifact_extractor = JsonPassTripletLLM(validate=False)
logging.info("Initializing GraphBuilder started")
# Initialize a graph builder pipeline
graph_builder = KnowledgeGraphBuilder(
    client,
    triplet_extractor=artifact_extractor,
    chunker=chunker
)
logging.info("Initializing GraphBuilder finished")
# Run building
logging.info("Start building graph")
knowledge_graph = graph_builder.build(text)
logging.info("Finish building graph")
# Save results
knowledge_graph.save_graph("./results/graph.gml").save_community_summary("./results/summary.json") # todo уточнить папку
logging.info("Graph Saved")