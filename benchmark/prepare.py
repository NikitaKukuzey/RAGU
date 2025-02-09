import hydra
from openai import OpenAI

from benchmark.common import JSONDataset, create_dir, settings
from ragu.graph.graph_rag import GraphRag


client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
)


@hydra.main(version_base=None, config_path="./../configs", config_name="benchmark")
def main(config) -> None:
    """Основная функция запуска обработки с использованием Hydra."""

    source_path = config["source"]

    json_dataset = JSONDataset(source_path)

    # Создание графа
    graph_rag = GraphRag(config).build(json_dataset.get_documents(), client)
    checkpoint_dir = create_dir(source=source_path)
    print("Checkpoint directory:", checkpoint_dir)

    # Сохранение результатов
    graph_rag.save_graph(checkpoint_dir / "graph.gml")
    graph_rag.save_community_summary(checkpoint_dir / "community_summary.parquet")


if __name__ == "__main__":
    main()
