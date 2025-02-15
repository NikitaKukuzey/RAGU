import re
from tqdm import tqdm

from ragu import Generator
from ragu.common.llm import BaseLLM


@Generator.register("original_generator")
class OriginalGenerator(Generator):
    """
    An implementation of the Generator class that creates a final answer by interacting 
    with an external model. This implementation generates intermediate answers from community 
    summaries, filters them, sorts them, and finally generates a response based on these answers.
    """
    
    def __init__(self, class_name):
        """
        Initializes the generator with model information and system prompts.
        
        :param class_name: The class name (not used in this context but can be extended).
        """
        super().__init__()
    
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
        from ragu.utils.default_prompts.generation_prompt import generation_rating_prompt
        from ragu.utils.default_prompts.generation_prompt import generation_final_answer_prompt

        intermediate_answers = []
        for _, summary in tqdm(enumerate(community_summaries), desc="Inference: Getting global answers."):
            text = f"Query: {query} Summary: {summary}"
            answer = client.generate(text, generation_rating_prompt)
            answer_parts = str(answer).split('<|>')
            
            # Skip if the answer doesn't match expected format
            if len(answer_parts) != 2:
                continue 

            digits_only = re.sub(r'\D', '', answer_parts[0])

            # TODO: try something better...
            if digits_only == "":
                rating = 3
            else:
                rating = int(digits_only)
            response = answer_parts[1]

            # Skip answers with a rating less than 3
            if rating < 3:
                continue  

            intermediate_answers.append((rating, response))

        intermediate_answers = sorted(intermediate_answers, key=lambda x: x[0], reverse=True)

        text = f"Intermediate answers: {intermediate_answers}"
        final_answer = client.generate(text, generation_final_answer_prompt)
        return final_answer