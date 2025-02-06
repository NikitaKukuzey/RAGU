from typing import Any, List, Optional
import opennre
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

    def extract_entities_and_relationships(self, text: str, client: Any) -> List[Any]:
        """
        Uses an LLM to extract entities and relationships from the input text.

        :param text: The input text to process.
        :param client: External API client for LLM interaction.
        :return: A list of extracted triplets.
        """
        from ragu.utils.default_prompts.triplet_maker_prompts import tripler_system_prompts
        from ragu.utils.triplet_parser import parse_relations

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": tripler_system_prompts},
                {"role": "user", "content": text}
            ]
        )
        return parse_relations(response.choices[0].message.content)


@TripletExtractor.register("opennre")
class TripletNEREL(TripletExtractor):
    """
    Extracts triplets using the OpenNRE model.
    """
    
    def __init__(self, class_name: str, model_name: str) -> None:
        """
        Initializes the OpenNRE-based triplet extractor.

        :param class_name: Identifier for the extractor class.
        :param model_name: Name of the OpenNRE model.
        """
        super().__init__()
        self.model_name = model_name
        self.opennre_model = opennre.get_model(self.model_name)

    def extract_entities_and_relationships(self, text: str, client: Optional[Any] = None) -> Any:
        """
        Uses the OpenNRE model to extract entities and relationships from text.

        :param text: The input text to process.
        :param client: Not used in this implementation.
        :return: Extracted triplet data.
        """
        return self.opennre_model.infer({'text': text})[0]