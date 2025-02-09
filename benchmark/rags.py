import time
from benchmark.common import BaseRAG, BaseDataset, settings


class ClassicRAG(BaseRAG):
    def __init__(self,
                 dataset: BaseDataset,
                 client,
                 embedding_model_name: str = "ai-forever/FRIDA"):
        super().__init__(dataset, client)

        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(embedding_model_name)
        self.index = []
        self.chunks = []
        self.build_index()

    # Основная функция для выполнения запроса и получения ответа
    def generate(self, query):
        # documents = self.dataset.get_documents()
        # # Генерация сходства между запросом и документами с помощью CrossEncoder
        # scores = self.model.predict(
        #     [(query, doc) for doc in documents])

        import numpy as np
        encoded_query = self.model.encode(query, convert_to_tensor=True).cpu().detach().numpy()

        similarity = encoded_query @ np.array(self.index).T
        relevant_chunks = sorted(zip(similarity, self.chunks), key=lambda x: x[0], reverse=True)[:4]
        relevant_chunks = [x[1] for x in relevant_chunks]

        # # Генерация ответа с использованием клиента OpenAI в чат-режиме
        input_text = query + "\n" + "\n".join(relevant_chunks)
        
        # Используем OpenAI Chat API
        completion = self.client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )

        # Возвращаем сгенерированный ответ и найденные документы
        return completion.choices[0].message.content.strip(), relevant_chunks
    

    def get_chunks(self, document, max_chunk_size: int=512):
        from razdel import sentenize
        
        chunks: list[str] = []
        sentences = [chunk.text for chunk in sentenize(document)]
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
    

    def get_embeddings(self, texts: list[str]):
        return [self.model.encode(text, convert_to_tensor=True).cpu().detach().numpy() for text in texts]

    def build_index(self):
        documents = self.dataset.get_documents()
        for document in documents:
            chunks = self.get_chunks(document)
            embeddings = self.get_embeddings(chunks)

            self.chunks.extend(chunks)
            self.index.extend(embeddings)

class RAGU(BaseRAG):
    def __init__(self, dataset, client, config, gml_path, summary_path):
        super().__init__(dataset, client)

        from ragu.graph.graph_rag import GraphRag

        self.graph_rag = GraphRag(config)
        self.graph_rag.load_knowledge_graph(
            path_to_graph=gml_path,
            path_to_community_summary=summary_path
        )

    def generate(self, query):
        answer, context = self.graph_rag.get_response(query, self.client)
        return answer, [c[1] for c in context]


class JustLLM(BaseRAG):
    def __init__(self, dataset, client):
        super().__init__(dataset, client)

    def generate(self, query):
        completion = self.client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content.strip(), ""
