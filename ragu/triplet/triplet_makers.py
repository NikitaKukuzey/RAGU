import opennre

from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    def __init__(
        self,
        class_name: str,
        model_name: str,
        system_prompt: str
    ):
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def extract_entities_and_relationships(self, text, client):
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content


@TripletExtractor.register("opennre")
class TripletNEREL(TripletExtractor):
    def __init__(
        self,
        class_name: str,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.opennre_model = opennre.get_model(self.model_name)

    def extract_entities_and_relationships(self, text):
        result = self.opennre_model.infer({'text': text})[0]
        return result
