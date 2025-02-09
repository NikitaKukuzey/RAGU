import json
import os

import hydra
from openai import OpenAI
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings

from benchmark.common import JSONDataset, settings, evaluate_rag, create_dir, init_logging
from benchmark.rags import JustLLM, ClassicRAG, RAGU
from benchmark.metrics import (
    BertScore,
    LLMJudgeEvaluator,
    ResponseRelevancyMetric,
    FaithfulnessMetric,
)

logger = init_logging()


def safe_evaluate(name, rag_instance, metrics, dataset, results):
    results[name] = evaluate_rag(
        rag=rag_instance, metrics=metrics, dataset=dataset)
    logger.info(f"{name} evaluation successful: {results[name]}")


test_set_json = {
    "chegeka": {
        "gml": "./benchmark/checkpoints/xxx/graph.gml",
        "summary": "./benchmark/checkpoints/xxx/community_summary.parquet"
    },
    "fairy-tails": {
        "gml": "./benchmark/checkpoints/extended_2025-02-09_06-53-22_yezXqu/graph.gml",
        "summary": "./benchmark/checkpoints/extended_2025-02-09_06-53-22_yezXqu/community_summary.parquet"
    }
}


client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
)


@hydra.main(version_base=None, config_path="./../configs", config_name="benchmark")
def main(config):
    """Главная функция, запускающая бенчмарк"""

    json_dataset = JSONDataset("./benchmark/data/fairy-tails/extended.json")
    json_dataset.dataset = json_dataset.dataset
    results = {}

    # === Инициализация моделей для RAGAS ===
    os.environ['OPENAI_API_KEY'] = settings.llm_api_key
    ragas_llm = llm_factory(
        model=settings.llm_model_name,
        base_url=settings.llm_base_url
    )
    ragas_embeddings = HuggingFaceEmbeddings(
        model_name="ai-forever/FRIDA",
        model_kwargs={
            "device": "cuda"
        },
    )

    metrics = [
        ResponseRelevancyMetric(
            llm=ragas_llm,
            embeddings=ragas_embeddings
        ),
        FaithfulnessMetric(
            llm=ragas_llm,
        ),
        BertScore(),
        LLMJudgeEvaluator(client),

    ]

    # === Оценка LLM ===
    # safe_evaluate(
    #     name="llm",
    #     rag_instance=JustLLM(json_dataset, client=client),
    #     metrics=metrics,
    #     dataset=json_dataset,
    #     results=results
    # )

    # === Оценка классического RAG ===
    safe_evaluate(
        name="classic",
        rag_instance=ClassicRAG(json_dataset, client=client),
        metrics=metrics,
        dataset=json_dataset,
        results=results
    )

    # === Оценка ragu ===
    # safe_evaluate(
    #     name="ragu",
    #     rag_instance=RAGU(dataset=json_dataset,
    #                       client=client,
    #                       config=config,
    #                       gml_path=test_set_json["fairy-tails"]["gml"],
    #                       summary_path=test_set_json["fairy-tails"]["summary"]
    #                       ),
    #     metrics=metrics,
    #     dataset=json_dataset,
    #     results=results
    # )

    logger.info(f"Results: {results}")
    target_name = "result-metrics.json"
    checkpoint_dir = create_dir(target_name)

    with open(checkpoint_dir / target_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Result metrics: {results}")
    logger.info(f"Saved in: {checkpoint_dir / target_name}")


if __name__ == "__main__":
    main()
