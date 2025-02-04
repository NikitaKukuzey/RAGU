from ragu.generate.base_generator import Generator


Generator.register("original-generator")
class OriginalGenerator(Generator):
    def __init__(self, config):
        super().__init__(config)

    def generate_final_answer(self, community_summaries, query, client):
        intermediate_answers = []
        for index, summary in enumerate(community_summaries):
            print(f"Summary index {index} of {len(community_summaries)}:")
            response = client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system",
                        "content": self.config.generator.original.summary_system_prompt},
                    {"role": "user", "content": f"Query: {query} Summary: {summary}"}
                ]
            )
            print("Intermediate answer:", response.choices[0].message.content)
            intermediate_answers.append(
                response.choices[0].message.content)

        final_response = client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {"role": "system",
                 "content": self.config.generator.original.final_system_prompt},
                {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
            ]
        )
        final_answer = final_response.choices[0].message.content
        return final_answer
