from ragu.generate.base_generator import Generator


@Generator.register("original_generator")
class OriginalGenerator(Generator):
    def __init__(self, class_name, model_name: str, system_prompt: str):
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate_final_answer(self, query, community_summaries, client):
        intermediate_answers = []
        for index, summary in enumerate(community_summaries):
            print(f"Summary index {index} of {len(community_summaries)}:")
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Query: {query} Summary: {summary}"}
                ]
            )
            print("Intermediate answer:", response.choices[0].message.content)
            intermediate_answers.append(
                response.choices[0].message.content
            )

        final_response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
            ]
        )
        final_answer = final_response.choices[0].message.content
        return final_answer
