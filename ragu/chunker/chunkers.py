from typing import List
import codecs
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from razdel import sentenize

from sentence_transformers import SentenceTransformer

from ragu.chunker.base_chunker import Chunker


@Chunker.register("simple")
class SimpleChunker(Chunker):
    def __init__(self, class_name: str, max_chunk_size: int, overlap: int):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def get_chunks(self, documents: list):
        chunks = []
        for document in documents:
            for i in range(0, len(document), self.max_chunk_size - self.overlap):
                chunk = document[i : i + self.max_chunk_size]
                chunks.append(chunk)
        return chunks


@Chunker.register("semantic")
class SemanticTextChunker(Chunker):
    def __init__(
            self, 
            class_name: str, 
            model_name: str, 
            max_chunk_size: int
        ):
        super().__init__()

        self.model = SentenceTransformer(model_name).to(self.get_device())
        self.model.eval()
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def split_text_by_chunks(self, text: List[str]) -> List[str]:
        return [mini_chunk.text for mini_chunk in list(sentenize(text))]

    def calculate_document_embedding(self, document: str) -> np.ndarray:
        embeddings = self.model.encode(document, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def compute_similarities(self, chunks: List[str]) -> np.ndarray:
        if len(chunks) < 2:
            return torch.tensor([])

        embeddings = np.vstack(
            [self.calculate_document_embedding('search_document: ' + chunk) for chunk in chunks]
        )
        return torch.tensor(
            [(embeddings[idx : (idx + 1)] @ embeddings[(idx + 1) : (idx + 2)].T).tolist()[0]
                for idx in range(len(chunks) - 1)
            ]
        ).reshape((len(chunks) - 1,))

    def join_chunks_by_semantics(
            self, 
            chunks: List[str], 
            similarities: np.ndarray
        ) -> List[str]:
        
        if len(chunks) < 2:
            return chunks
  
        n_tokens = len(self.model.tokenize('search_document: ' + '\n'.join(chunks))['input_ids'])

        if n_tokens <= self.max_chunk_size:
            return ['\n'.join(chunks)]

        min_similarity_idx = torch.argmin(similarities)

        if min_similarity_idx == 0:
            res = [chunks[0]]
        else:
            res = self.join_chunks_by_semantics(
                chunks[:(min_similarity_idx + 1)],
                similarities[:min_similarity_idx]
            )
        if min_similarity_idx == (len(chunks) - 2):
            res += [chunks[-1]]
        else:
            res += self.join_chunks_by_semantics(
                chunks[(min_similarity_idx + 1):], 
                similarities[(min_similarity_idx + 1):]
            )
        return res
    
    def get_chunks(self, text: list[str]) -> List[str]:
        # text = self.load_text(file_path)
        chunks = self.split_text_by_chunks(text)
        similarities = self.compute_similarities(chunks)
        return self.join_chunks_by_semantics(chunks, similarities)
