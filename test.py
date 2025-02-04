import hydra
from ragu import Chunker


@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def main(config):
    chunker = Chunker.get_by_name(**config.chunker)
    # chunker = SemanticTextChunker(config.chunker)

    chunks = chunker.get_chunks("data/test_text.txt")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")


if __name__ == "__main__":
    main()