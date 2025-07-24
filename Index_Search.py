from ragu.utils.io_utils import read_text_from_files, read_qa_from_chegeka
from ragu.common.llm import RemoteLLM, LocalLLM
from ragu.graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph

from ragu.search_engine.local_search import LocalSearchEngine
from ragu.common.embedder import STEmbedder
from ragu.chunker.chunkers import SmartSemanticChunker, JsonPassChunker
from ragu.triplet.triplet_makers import TripletLLM, JsonTripletLLM
from ragu.common.index import Index

from ragu.LLMjudge import compare_llm_answers_with_llm

import pandas as pd
import numpy as np
import json

PATH_TO_CHEGEKA="./chegeka"

client = LocalLLM("yandex/YandexGPT-5-Lite-8B-instruct")

knowledge_graph = KnowledgeGraph(path_to_graph="path/to/graph", path_to_community_summary="path/to/community") #todo уточнить папку

embedder = STEmbedder("SentenceTransformer", trust_remote_code=True)
index = Index(embedder=embedder)
index.make_index(knowledge_graph)

local_search = LocalSearchEngine(
    client,
    knowledge_graph,
    embedder,
    index
)

qas = read_qa_from_chegeka(PATH_TO_CHEGEKA + '/' + 'qa.json')

answers = []
outputs = []
dct_to_save = []

for el in qas:
    out = local_search.query(el[0])
    answers.append(el[1])
    outputs.append(out)
    dct_to_save.append({"question": el[0], "answer": el[1], "output": out})

with open(PATH_TO_CHEGEKA + '/view_answers.json', 'w', encoding='utf-8') as f:
    json.dump(dct_to_save, f, ensure_ascii=False, indent=4)

score = compare_llm_answers_with_llm(answers, outputs)
print(score)