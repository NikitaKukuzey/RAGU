import re
import logging
from tqdm import tqdm

from ragu import Generator
from ragu.common.batch_generator import BatchGenerator
from ragu.common.llm import BaseLLM


@Generator.register("original_generator")
class OriginalGenerator(Generator):
    """
    An implementation of the Generator class that creates a final answer by interacting 
    with an external model. This implementation generates intermediate answers from community 
    summaries, filters them, sorts them, and finally generates a response based on these answers.
    """
    
    def __init__(self, class_name, batch_size: int):
        """
        Initializes the generator with model information and system prompts.
        
        :param class_name: The class name (not used).
        """
        super().__init__()
        self.batch_size = batch_size

    def generate_final_answer(self, query, community_summaries, client: BaseLLM, *args, **kwargs):
        """
        Generates a final answer by obtaining intermediate answers from the model, 
        filtering and sorting them based on a rating, 
        and using the sorted answers to generate the final response.

        :param query: The query to generate a final answer for.
        :param community_summaries: The community summaries to be processed.
        :param client: The client responsible for making API requests to the model.
        :return: The final answer generated after processing intermediate answers.
        """
        from ragu.utils.default_prompts.generation_prompt import (
            generation_rating_prompt,
            generation_final_answer_prompt
        )
        batch_generator = BatchGenerator(community_summaries, batch_size=self.batch_size)

        intermediate_answers = []
        for batch in tqdm(
                batch_generator.get_batches(),
                desc="Inference: getting global answers.",
                total=len(batch_generator)
        ):
            texts = [f"Query: {query} Summary: {summary}" for summary in batch]
            answers = client.generate(texts, generation_rating_prompt)

            for answer in answers:
                answer_parts = str(answer).split('<|>')

                if len(answer_parts) != 2:
                    continue

                digits_only = re.sub(r'\D', '', answer_parts[0])

                if digits_only == "" or digits_only is None:
                    logging.error(f"Rating is not a number. Text: {answer}")

                rating = int(digits_only) if digits_only else 3
                response = answer_parts[1]

                if rating < 3:
                    continue

                intermediate_answers.append((rating, response))

        intermediate_answers = sorted(intermediate_answers, key=lambda x: x[0], reverse=True)
        text = f"Intermediate answers: {intermediate_answers}"
        final_answer = client.generate(text, generation_final_answer_prompt)
        return final_answer