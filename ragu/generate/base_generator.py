from abc import ABC, abstractmethod

from environment import Registrable


class Generator(ABC, Registrable):
    def __init__(self):
        ...
        # self.client = client
        # self.config = config

    @abstractmethod
    def generate_final_answer(self, community_summaries, query):
        """
        Генерация финального ответа на основе нескольких резюме сообществ и запроса.
        """
        pass
    
    def __call__(self, community_summaries, query):
        self.generate_final_answer(community_summaries, query)
