import torch
import numpy as np
from typing import List

from razdel import sentenize
from sentence_transformers import SentenceTransformer

from ragu.chunker.base_chunker import Chunker


@Chunker.register("simple")
class SimpleChunker(Chunker):
    """
    A simple chunker that splits text into fixed-size overlapping chunks.
    """
    
    def __init__(self, class_name: str, max_chunk_size: int, overlap: int) -> None:
        """
        Initializes the simple chunker.

        :param class_name: Identifier for the chunker class.
        :param max_chunk_size: Maximum chunk size in characters.
        :param overlap: Number of overlapping characters between consecutive chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def get_chunks(self, documents: List[str]) -> List[str]:
        """
        Splits documents into fixed-size overlapping chunks.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        chunks = []
        for document in documents:
            for i in range(0, len(document), self.max_chunk_size - self.overlap):
                chunk = document[i : i + self.max_chunk_size]
                chunks.append(chunk)
        return chunks


@Chunker.register("semantic")
class SemanticTextChunker(Chunker):
    """
    A semantic chunker that splits text based on sentence boundaries and semantic similarity.
    """
    
    def __init__(self, class_name: str, model_name: str, max_chunk_size: int) -> None:
        """
        Initializes the semantic chunker.

        :param class_name: Identifier for the chunker class.
        :param model_name: Name of the sentence transformer model.
        :param max_chunk_size: Maximum chunk size in tokens.
        """
        super().__init__()
        self.model = SentenceTransformer(model_name).to(self.get_device())
        self.model.eval()
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def get_device() -> str:
        """
        Determines the appropriate device for model execution.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def split_text_by_chunks(self, text: str) -> List[str]:
        """
        Splits text into sentence-based chunks using Razdel.

        :param text: Input text.
        :return: List of sentence-based chunks.
        """
        return [chunk.text for chunk in sentenize(text)]

    def calculate_document_embedding(self, document: str) -> np.ndarray:
        """
        Computes the embedding of a given document using the sentence transformer model.

        :param document: Input document.
        :return: Document embedding as a NumPy array.
        """
        embeddings = self.model.encode(document, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def compute_similarities(self, chunks: List[str]) -> np.ndarray:
        """
        Computes pairwise cosine similarities between consecutive chunks.

        :param chunks: List of sentence-based chunks.
        :return: Similarity scores as a NumPy array.
        """
        if len(chunks) < 2:
            return np.array([])

        embeddings = np.vstack(
            [self.calculate_document_embedding(chunk) for chunk in chunks]
        )
        similarities = np.array([
            np.dot(embeddings[i], embeddings[i + 1])
            for i in range(len(chunks) - 1)
        ])
        return similarities

    def join_chunks_by_semantics(self, chunks: List[str], similarities: np.ndarray) -> List[str]:
        """
        Merges chunks based on semantic similarity, ensuring chunk sizes do not exceed the limit.

        :param chunks: List of sentence-based chunks.
        :param similarities: Pairwise similarity scores between chunks.
        :return: List of semantically merged text chunks.
        """
        if len(chunks) < 2:
            return chunks

        n_tokens = len(self.model.tokenize(" ".join(chunks))["input_ids"])
        if n_tokens <= self.max_chunk_size:
            return [" ".join(chunks)]

        min_similarity_idx = np.argmin(similarities)
        return (
            self.join_chunks_by_semantics(
                chunks[: min_similarity_idx + 1], 
                similarities[:min_similarity_idx]
            )
            + self.join_chunks_by_semantics(
                chunks[min_similarity_idx + 1 :], 
                similarities[min_similarity_idx + 1 :]
            )
        )

    def get_chunks(self, documents: List[str]) -> List[str]:
        """
        Splits text into semantically coherent chunks.

        :param documents: List of input documents.
        :return: List of semantically split text chunks.
        """
        all_chunks = []
        for document in documents:
            chunks = self.split_text_by_chunks(document)
            similarities = self.compute_similarities(chunks)
            all_chunks.extend(self.join_chunks_by_semantics(chunks, similarities))
        return all_chunks
