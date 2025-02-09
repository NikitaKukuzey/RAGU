### Benchmark

This benchmark is designed to evaluate performance. It includes code for data processing, an API server for the prototype, and modules for metric calculation.

### 📁 Benchmark Structure

```
benchmark/
│── checkpoints/                # Folder for saved model checkpoints.
│   ├── fairy-tails_2025-02-07_17-47-04_wiYurO/  # Example directory with checkpoints for indexed graphs.
│── data/                       # Data for the benchmark.
│   ├── chegeka/                # "chegeka" dataset.
│   ├── fairy-tails/            # Synthetic dataset.
│── .env                        # Secrets.
│── __init__.py                 # Python package initialization file.
│── api.py                      # Module for API interaction.
│── prepare.py                   # Module for data preparation or preprocessing.
│── README.md                   
│── requirements-benchmark.in   # Dependency file for the benchmark.
│── requirements-benchmark.txt  # Compiled dependency file for the benchmark.
```

---

> When running the commands below, environment and dependency setup is performed automatically. Commands should be executed from the project root.

### 🛠 Graph Preparation

At this stage, we propose constructing a knowledge graph based on the specified dataset. The dataset is expected to have the following format:

```json
[
    {
        "instruction": <string - instruction on what the system should do>,
        "inputs": {
            "text": <string - question>,
        },
        "outputs": <string - reference answer>,
        "meta": { <additional information> },
        "wiki_articles": {"Bjørnstjerne Bjørnson": <document text> ...}
]
```

After loading the dataset, the graph indexing process will begin. Upon completion, two files will be saved in the specified checkpoint directory:  
- `graph.gml` - contains the graph  
- `summary.parquet` - summary of the identified communities  

```bash
make index-bench SOURCE=<path to dataset>
```

### 🚀 Running the Benchmark

```bash
make run-bench
```

### Data Corpus Preparation

Our benchmark uses two datasets:
1. [CheGeKa](https://mera.a-ai.ru/ru/tasks/8). We randomly selected 100 questions. For each question, we asked Claude-Sonnet-3.5 (with web search) to find relevant Wikipedia articles. This resulted in a set of questions along with relevant articles for building a graph-based knowledge base.
2. Additionally, we created a synthetic dataset by generating fairy tales on a given topic using DeepSeek-R1.
