from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.cross_encoder import CrossEncoder

from ragu.reranker.base_reranker import Reranker


@Reranker.register("dummy_reranker")
class DummyReranker(Reranker):
    """
    A dummy reranker that returns the input documents as-is without reranking.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the DummyReranker with arbitrary arguments.
        """
        super().__init__()

    def get_relevant_chunks(self, query: str, documents: List[str]) -> List[str]:
        """
        Returns the input documents unchanged.

        :param query: The input query (unused).
        :param documents: List of candidate documents.
        :return: The same list of documents.
        """
        return documents


@Reranker.register("cross_encoder_reranker")
class CrossEncoderReranker(Reranker):
    """
    Reranker that uses a CrossEncoder model to score document-query pairs.
    """
    
    def __init__(self, class_name: str, model_name: str, top_k: int = 10) -> None:
        """
        Initializes the CrossEncoder reranker.

        :param class_name: Identifier for the reranker class.
        :param model_name: Name of the CrossEncoder model.
        :param top_k: Number of top-ranked documents to return.
        """
        super().__init__()
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def get_relevant_chunks(self, query: str, documents: List[str]) -> List[str]:
        """
        Reranks documents based on relevance scores predicted by the CrossEncoder model.

        :param query: The input query.
        :param documents: List of candidate documents.
        :return: A ranked list of the top-k relevant documents.
        """
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        top_results = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )[:self.top_k]
        
        return [doc for doc, _ in top_results]


def _get_relevance_score(
        query,
        summary_index,
        bm25,
        embedder,
        bm25_weight,
        semantic_weight
):
    tokenized_query = query.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    query_embedding = embedder.encode([query])

    cos_sim = cosine_similarity(query_embedding, summary_index)[0]
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)

    return bm25_weight * bm25_norm + semantic_weight * cos_sim_norm


@Reranker.register("hybrid_reranker_top_k")
class HybridRerankerTopK(Reranker):
    """
    Reranker that selects the top-k most relevant documents.
    """
    def __init__(
            self,
            class_name: str,
            bm25_weight: float=0.25,
            semantic_weight: float=0.75,
            top_k: int = 10
    ) -> None:
        """
        Initialize the Top-K hybrid reranker.

        :param bm25_weight: Weight assigned to BM25 scores.
        :param semantic_weight: Weight assigned to semantic similarity scores.
        :param top_k: Number of top documents to retrieve.
        """

        super().__init__()
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.top_k = top_k

    def get_relevant_chunks(
            self,
            query,
            documents,
            summary_index,
            bm25,
            embedder
    ):
        """
        Retrieve the top-k relevant documents.

        :param query: Search query string.
        :param documents: List of document texts.
        :param summary_index: Array representing document embeddings.
        :param bm25: BM25 retrieval model instance.
        :param embedder: Model used for generating semantic embeddings.
        :return: List of top-k relevant document texts.
        """
        score = _get_relevance_score(query, summary_index, bm25, embedder, self.bm25_weight, self.semantic_weight)
        top_indices = np.argsort(score)[::-1][:self.top_k]
        return [documents[i] for i in top_indices]


@Reranker.register("hybrid_reranker_threshold")
class HybridRerankerThreshold(Reranker):
    """
    Reranker that selects documents exceeding a relevance threshold.
    """
    def __init__(
            self,
            class_name: str,
            bm25_weight: float = 0.25,
            semantic_weight: float = 0.75,
            threshold: int =0.5
    ) -> None:
        """
        Initialize the threshold-based hybrid reranker.

        :param bm25_weight: Weight assigned to BM25 scores.
        :type bm25_weight: float
        :param semantic_weight: Weight assigned to semantic similarity scores.
        :type semantic_weight: float
        :param threshold: Minimum relevance score for document selection.
        :type threshold: float
        """
        super().__init__()
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.threshold = threshold

    def get_relevant_chunks(
            self,
            query: str,
            documents: List[str],
            summary_index: np.ndarray,
            bm25: BM25Okapi,
            embedder: object,
    ) -> List[str]:
        """
        Retrieve the top-k relevant documents.

        :param query: Search query string.
        :param documents: List of document texts.
        :param summary_index: Array representing document embeddings.
        :param bm25: BM25 retrieval model instance.
        :param embedder: Model used for generating semantic embeddings.
        :return: List of top-k relevant document texts.
        """
        score = _get_relevance_score(
            query,
            summary_index,
            bm25,
            embedder,
            self.bm25_weight,
            self.semantic_weight
        )
        top_indices = [i for i in range(len(score)) if score[i] > self.threshold]
        return [documents[i] for i in top_indices]