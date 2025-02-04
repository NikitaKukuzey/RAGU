from typing import List
import codecs
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from razdel import sentenize

from src.chunker.base_chunker import BaseChunker


class SimpleChunker(BaseChunker):
    def __init__(self, config):
        super().__init__(config)
        self.max_chunk_size = config.max_chunk_size

    def get_chunks(self, documents: list):
        chunks = []
        for document in documents:
            for i in range(0, len(document), self.chunk_size - self.chunk_overlap):
                chunk = document[i : i + self.chunk_size]
                chunks.append(chunk)
        return chunks


class SemanticTextChunker(BaseChunker):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(
            self.get_device()
        )
        self.model.eval()

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_text(self, file_path: str) -> List[str]:
        with codecs.open(file_path, mode="r", encoding="utf-8") as fp:
            return fp.read().strip()

    def split_text_by_chunks(self, text: List[str]) -> List[str]:
        return [mini_chunk.text for mini_chunk in list(sentenize(text))]

    def calculate_document_embedding(self, document: str) -> np.ndarray:
        inputs = self.tokenizer(
            ["search_document: " + document],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return (
            torch.nn.functional.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            .cpu()
            .numpy()
        )

    def compute_similarities(self, chunks: List[str]) -> np.ndarray:
        if len(chunks) < 2:
            return torch.tensor([])

        embeddings = np.vstack(
            [self.calculate_document_embedding(chunk) for chunk in chunks]
        )
        return torch.tensor(
            [(embeddings[idx : (idx + 1)] @ embeddings[(idx + 1) : (idx + 2)].T).tolist()[0]
                for idx in range(len(chunks) - 1)
            ]
        ).reshape((len(chunks) - 1,))

    def join_chunks_by_semantics(self, chunks: List[str], similarities_between_neighbors: np.ndarray) -> List[str]:
        if len(chunks) < 2:
            return chunks
  
        n_tokens = len(self.tokenizer.tokenize('search_document: ' + '\n'.join(chunks)))

        if n_tokens <= self.config.max_chunk_size:
            return ['\n'.join(chunks)]
        
        min_similarity_idx = torch.argmin(similarities_between_neighbors)

        if min_similarity_idx == 0:
            res = [chunks[0]]
        else:
            res = self.join_chunks_by_semantics(
                chunks[:(min_similarity_idx + 1)],
                similarities_between_neighbors[:min_similarity_idx]
            )
        if min_similarity_idx == (len(chunks) - 2):
            res += [chunks[-1]]
        else:
            res += self.join_chunks_by_semantics(
                chunks[(min_similarity_idx + 1):], 
                similarities_between_neighbors[(min_similarity_idx + 1):]
            )
        return res
    
    def get_chunks(self, file_path: str) -> List[str]:
        text = self.load_text(file_path)
        chunks = self.split_text_by_chunks(text)
        similarities = self.compute_similarities(chunks)
        return self.join_chunks_by_semantics(chunks, similarities)
