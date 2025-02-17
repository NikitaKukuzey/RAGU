from pyexpat.errors import messages

from transformers import pipeline
from openai import OpenAI


class BaseLLM:
    def __init__(self):
        ...

    def generate(self, *args, **kwargs):
        ...


class LocalLLM(BaseLLM):
    def __init__(self, model_name: str, *args, **kwargs):
        self.pipe = pipeline("text-generation", model_name, **kwargs)
        super().__init__()

    def generate(self, query: str, system_prompt: str, *args, **kwargs):
        message = '\n'.join([system_prompt, query])
        result = self.pipe(message, **kwargs)
        return result[0]['generated_text'] if isinstance(result, list) else result


class RemoteLLM(BaseLLM):
    def __init__(self, model_name: str, base_url: str, api_token: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_token, **kwargs)

    def generate(self, query: str, system_prompt: str, model_name: str=None, *args, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content if response.choices else ""

