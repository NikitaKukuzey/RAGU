from base_chunker import BaseChunker

from abc import ABC, abstractmethod
from typing import List, Tuple
import os
import codecs
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


class SimpleChunker(BaseChunker):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']

    def get_chunks(self, documents: list):
        chunks = []
        for document in documents:
            for i in range(0, len(document), self.chunk_size - self.chunk_overlap):
                chunk = document[i:i + self.chunk_size]
                chunks.append(chunk)
        return chunks


class SemanticTextChunker(BaseChunker):
    def __init__(
            self,
            config: dict,
            tokenizer: RobertaTokenizer,
            model: RobertaModel
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model = model.to(self.get_device())

    @staticmethod
    def get_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_text(self, file_path: str) -> List[str]:
        with codecs.open(file_path, mode='r', encoding='utf-8') as fp:
            return [line.strip() for line in fp.readlines()]

    def split_text_by_chunks(self, text: List[str]) -> List[str]:
        chunks = []
        new_chunk = ''
        empty_count = 0

        for line in text:
            if not line:
                empty_count += 1
                if empty_count >= 3:
                    empty_count = 0
                    new_chunk = new_chunk.strip()
                    if new_chunk:
                        chunks.append(new_chunk)
                    new_chunk = ''
            else:
                empty_count = 0
                new_chunk += (line + '\n')

        if new_chunk.strip():
            chunks.append(new_chunk.strip())

        return chunks

    def calculate_document_embedding(self, document: str) -> np.ndarray:
        inputs = self.tokenizer(
            ['search_document: ' + document],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.nn.functional.normalize(
            outputs.last_hidden_state[:, 0],
            p=2,
            dim=1
        ).cpu().numpy()

    def compute_similarities(
            self,
            chunks:
            List[str]
    ) -> np.ndarray:
        embeddings = np.vstack(
            [self.calculate_document_embedding(chunk) for chunk in chunks]
        )
        return np.array(
            [(embeddings[idx] @ embeddings[idx + 1].T)[0] for idx in range(len(chunks) - 1)]
        )

    def join_chunks_by_semantics(
            self,
            chunks: List[str],
            similarities: np.ndarray
    ) -> List[str]:
        if len(chunks) < 2:
            return chunks

        max_text_len = 512
        num_tokens = len(self.tokenizer.tokenize('search_document: ' + '\n'.join(chunks)))

        if num_tokens <= max_text_len:
            return ['\n'.join(chunks)]

        min_sim_idx = np.argmin(similarities)

        left_chunks = self.join_chunks_by_semantics(chunks[:min_sim_idx + 1], similarities[:min_sim_idx])
        right_chunks = self.join_chunks_by_semantics(chunks[min_sim_idx + 1:], similarities[min_sim_idx + 1:])

        return left_chunks + right_chunks

    def get_chunks(self, file_path: str) -> List[str]:
        text = self.load_text(file_path)
        chunks = self.split_text_by_chunks(text)
        similarities = self.compute_similarities(chunks) if len(chunks) > 1 else np.array([])
        return self.join_chunks_by_semantics(chunks, similarities)

