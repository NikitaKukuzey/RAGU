from tqdm import tqdm
from typing import List, Dict

from ragu import Generator
from ragu.common.batch_generator import BatchGenerator
from ragu.common.llm import BaseLLM
from ragu.utils.parse_json_output import extract_json
from ragu.utils.default_prompts.generation_prompt import generator_prompts


@Generator.register("original_generator")
class OriginalGenerator(Generator):
    """
    An implementation of the Generator class that creates a final answer by interacting 
    with an external model. This implementation generates intermediate answers from community 
    summaries, filters them, sorts them, and finally generates a response based on these answers.
    """
    
    def __init__(self, batch_size: int):
        """
        Initializes the generator with model information and system prompts.
        """
        super().__init__()
        self.batch_size = batch_size

    def generate_final_answer(
            self,
            query: str,
            community_summaries: List[str],
            client: BaseLLM,
            *args,
            **kwargs
    ) -> str:
        """
        Generates a final answer using intermediate responses filtered and ranked based on relevance.

        :param query: The input query.
        :param community_summaries: Summaries used for context.
        :param client: LLM client for generating responses.
        :return: The final generated answer.
        """
        batch_generator = BatchGenerator(community_summaries, batch_size=self.batch_size)
        raw_intermediate_answers = self._generate_intermediate_answers(query, batch_generator, client)
        intermediate_answers = self._process_intermediate_answers(raw_intermediate_answers)
        final_answer = self._generate_final_response(query, intermediate_answers, client)

        return final_answer

    def _generate_intermediate_answers(self, query: str, batch_generator: BatchGenerator, client: BaseLLM) -> List[str]:
        """
        Generates intermediate answers for each batch of community summaries.

        :param query: The input query.
        :param batch_generator: BatchGenerator instance to iterate over batches.
        :param client: LLM client for generating responses.
        :return: List of raw intermediate answers.
        """
        raw_intermediate_answers = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Inference: generating intermediate answers",
                total=len(batch_generator)
        ):
            context_text = self._format_context(batch)
            prompt_text = f"Запрос: {query}\n\nКонтекст:\n{context_text}\n\n"
            raw_answers = client.generate(prompt_text, generator_prompts["generate_intermediate_responses"])
            raw_intermediate_answers.extend(self._ensure_list(raw_answers))

        return raw_intermediate_answers

    def _format_context(self, batch: List[str]) -> str:
        """
        Formats the context text for a given batch of summaries.

        :param batch: List of summaries in the current batch.
        :return: Formatted context text.
        """
        return "\n".join(f"{i + 1}. {summary}" for i, summary in enumerate(batch))

    def _ensure_list(self, raw_answers: str | List[str]) -> List[str]:
        """
        Ensures the raw answers are always returned as a list.

        :param raw_answers: Raw answers from the LLM client.
        :return: List of raw answers.
        """
        return [raw_answers] if isinstance(raw_answers, str) else raw_answers

    def _process_intermediate_answers(self, raw_intermediate_answers: List[str]) -> List[dict]:
        """
        Processes raw intermediate answers by extracting JSON, filtering, and sorting.

        :param raw_intermediate_answers: List of raw intermediate answers.
        :return: List of processed intermediate answers.
        """
        intermediate_answers: List[Dict] = []
        for answer in raw_intermediate_answers:
            extracted_data = extract_json(answer)
            if extracted_data:
                intermediate_answers.extend(extracted_data["points"])

        # Filter and sort intermediate answers by score
        intermediate_answers = [
            ans for ans in intermediate_answers if ans.get("score", 0) > 0
        ]
        intermediate_answers.sort(key=lambda x: x.get("score", 0), reverse=True)
        return intermediate_answers

    def _generate_final_response(self, query: str, intermediate_answers: List[dict], client: BaseLLM) -> str:
        """
        Generates the final response using the processed intermediate answers.

        :param intermediate_answers: list of processed intermediate answers.
        :param client: LLM client for generating responses.
        :return: Final generated answer.
        """
        formatted_answers = "\n".join(
            f"Ответ: {ans['description']}" for ans in intermediate_answers
        )
        context = f"Вопрос:{query}\n\nПромежуточные ответы:\n{formatted_answers}"
        return client.generate(context, generator_prompts["generate_final_answer"])
