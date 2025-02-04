import hydra
from hydra.utils import instantiate
from openai import OpenAI
from pathlib import Path

from ragu.graph.build import GraphBuilder
from ragu import TripletExtractor, Generator, Chunker
from environment import settings


def read_text_from_files(directory: Path, file_extensions=None):
    """
    Чтение текста из всех файлов в директории и её поддиректориях.

    :param directory: Путь к директории для обхода (объект Path).
    :param file_extensions: Список расширений файлов для фильтрации (например, ['.txt', '.md']).
    :return: Список строк, каждая из которых — содержимое файла.
    """
    texts = []

    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Не удалось прочитать {file_path}: {e}")

    return texts


client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
)


@hydra.main(version_base=None, config_path="./configs", config_name="pipeline")
def main(config):
    print(config)
    chunker = Chunker.get_by_name(config.chunker.class_name)
    triplet = TripletExtractor.get_by_name(config.triplet.class_name)
    generator = Generator.get_by_name(config.generator.class_name)

    graph_builder = GraphBuilder(
        client=client,
        config=config)
    generator = instantiate(config.generator)

    texts = read_text_from_files(Path("./data/"), ['.txt', '.md', '.py'])
    print(chunker, type(chunker))

    chunks = []
    for i, text in enumerate(texts):
        chunks.append(chunker(text))

    print(chunks)

    elements = triplet(chunks, client=client)

    G, summaries = graph_builder(elements)

    answer = generator(G, client=client)

    print(answer)


if __name__ == "__main__":
    main()
