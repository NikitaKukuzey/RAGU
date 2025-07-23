import json
import os
from llama_index.core import Document


class RAGDataset:
    def __init__(self, qa_file_path, documents_file_path):
        if not os.path.exists(qa_file_path):
            raise FileNotFoundError(f"Файл {qa_file_path} не найден.")

        if not os.path.exists(documents_file_path):
            raise FileNotFoundError(f"Файл {documents_file_path} не найден.")

        with open(qa_file_path, 'r', encoding='utf-8') as file:
            self.qa_dataset = json.load(file)

        with open(documents_file_path, 'r', encoding='utf-8') as file:
            self.documents_dataset = json.load(file)

    def get_documents(self, use_llama_index_type: bool=False) -> list:
        documents = [doc['page_content'] for doc in self.documents_dataset]
        if use_llama_index_type:
            documents = [Document(text=doc) for doc in documents]
        return documents
    
    def get_number_of_queries(self) -> int:
        return len(self.qa_dataset)

    def get_samples(self):
        for entry in self.qa_dataset:
            yield entry["inputs"]["text"], entry["outputs"]

