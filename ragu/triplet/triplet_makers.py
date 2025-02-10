from typing import Any, List

from tqdm import tqdm

from ragu.common.settings import settings
from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    Extracts triplets using a large language model (LLM).
    """
    
    def __init__(self, class_name: str, model_name: str, system_prompt: str) -> None:
        """
        Initializes the LLM-based triplet extractor.

        :param class_name: Identifier for the extractor class.
        :param model_name: Name of the LLM model.
        :param system_prompt: System prompt used for LLM-based extraction.
        """
        super().__init__()
        self.model_name = model_name

    def extract_entities_and_relationships(self, elements: List[str], client: Any) -> List[Any]:
        """
        Uses an LLM to extract entities and relationships from the input text.

        :param text: The input text to process.
        :param client: External API client for LLM interaction.
        :return: A list of extracted triplets.
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import tripler_system_prompts
        from ragu.utils.triplet_parser import parse_relations

        results = []
        raw_output = []
        for text in tqdm(elements, desc='Index create: extract entities'):
            print(text)
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": tripler_system_prompts},
                    {"role": "user", "content": text}
                ]
            )
            raw_relations = response.choices[0].message.content

            response = parse_relations(raw_relations)
            results.extend(response)

            raw_output.append({'relations': raw_relations, 'chunk': text})
            
        return results, raw_output


