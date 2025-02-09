import os
import json
from pathlib import Path
import datetime
import random
import string
from abc import ABC, abstractmethod
import logging
import time

from tqdm import tqdm

from ragu.common.settings import Settings


# === Настройка логирования ===
def init_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('./benchmark/run.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = init_logging()


def create_dir(source: str) -> Path:
    """Создает уникальную директорию для чекпоинтов, используя имя файла, временную метку и случайный идентификатор."""

    file_name = Path(source).stem  # Извлекаем имя файла без расширения
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    random_part = ''.join(random.choices(
        string.ascii_letters + string.digits, k=6))

    checkpoint_dir = Path("./benchmark/checkpoints") / \
        f"{file_name}_{timestamp}_{random_part}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir


# === Интерфейс для датасета ===


class BaseDataset(ABC):
    def __init__(self):
        pass

    def get_documents(self) -> list[str]:
        pass

# === Базовые классы ===


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, query, prediction, reference, context):
        pass

    @abstractmethod
    def __name__(self):
        pass


class BaseRAG(ABC):
    def __init__(self, dataset: BaseDataset, client):
        self.dataset = dataset
        self.client = client

    @abstractmethod
    def generate(self, *args, **kwds) -> tuple[str, list[str]]:
        pass


# === Датасет для загрузки json ===


class JSONDataset(BaseDataset):
    dataset: dict = {}

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")

        with open(file_path, 'r', encoding='utf-8') as file:
            self.dataset = json.load(file)

    def get_documents(self):
        documents = {}
        for entry in self.dataset:
            documents.update(entry.get("wiki_articles", {}))

        return list(documents.values())

    def get_samples(self):
        for entry in self.dataset:
            query = entry["instruction"].format(**entry["inputs"])
            yield query, entry["outputs"]

# === Метод для оценки ===


def evaluate_rag(rag: BaseRAG,
             metrics: list[BaseMetric],
             dataset: BaseDataset):
    results = {"query": [], "prediction": [], "reference": [], "context": []}
    total_generate_time = 0  # Время работы rag.generate

    for query, reference in tqdm(dataset.get_samples(), desc="Evaluating", unit=" sample"):
        gen_start = time.perf_counter()
        prediction, context = rag.generate(query)
        gen_end = time.perf_counter()
        total_generate_time += gen_end - gen_start
        results["query"].append(query)
        results["prediction"].append(prediction)
        results["reference"].append(reference)
        results["context"].append(context)
        
    avg_generate_time = total_generate_time / len(results) if results else 0
    logger.info(f"Среднее время генерации: {avg_generate_time}")
    return {
        metric.__class__.__name__: metric.compute(**results)
        for metric in metrics
    }


settings = Settings.from_env("./benchmark/.env")
