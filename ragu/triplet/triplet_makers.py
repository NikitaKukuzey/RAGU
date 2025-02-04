from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def extract_entities_and_relationships(self, text, client):
        response = self.client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {"role": "system", "content": self.config.triplet.system_prompt},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
