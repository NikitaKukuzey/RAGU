from typing import List

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
