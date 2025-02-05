from abc import ABC, abstractmethod

from environment import Registrable


class TripletExtractor(ABC, Registrable):
    def __init__(self, config: dict=None):
        self.config = config

    @abstractmethod
    def extract_entities_and_relationships(self, text):
        """
        Абстрактный метод для извлечения сущностей и отношений из текста.
        Должен быть реализован в подклассах.
        """
        pass

    def __call__(self, elements: list[str], client):
        """
        Обрабатывает список элементов и извлекает сущности и отношения для каждого из них.
        """
        results = []
        for index, element in enumerate(elements):
            print(f"Processing element {index + 1} of {len(elements)}")
            result = self.extract_entities_and_relationships(element, client)
            print("Extraction result:", result)
            results.append(result)
        return results
