import logging

from ragu.common.index import Index
from ragu.common.llm import BaseLLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.utils.ragu_utils import TokenTruncation
from ragu.utils.default_prompts.search_engine_query_prompts import (
    global_search_engine_prompt,
    global_search_meta_prompt,
    system_prompt
)

from ragu.search_engine.search_functional import global_search_default_extractor
from ragu.utils.parse_json_output import (
    extract_json,
    create_text_from_community
)


class GlobalSearchEngine(BaseEngine):
    def __init__(
            self,
            client: BaseLLM,
            index: Index,
            max_context_length: int = 30_000,
            tokenizer_backend: str = "tiktoken",
            tokenizer_model: str = "gpt-4",
            extractor=global_search_default_extractor,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.client = client

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length
        )

        self.extractor = extractor

    async def search(self, query, *args, **kwargs):
        responses: list[str] = []
        for community_cluster_id in await self.index.communities_kv_storage.all_keys():
            try:
                community = await self.index.communities_kv_storage.get_by_id(community_cluster_id)
                responses.append(await self.get_meta_responses(query, create_text_from_community(community)))
            except ValueError as e:
                logging.warning(e)

        responses = list(filter(lambda x: x.get("rating", 0) > 0, responses))
        responses: list[dict] = sorted(responses, key=lambda x: x.get("rating", 0), reverse=True)

        return "\n".join([r.get("response", "") for r in responses])

    def build_index(self):
        pass

    async def get_meta_responses(self, query: str, context: str) -> str:
        output = self.client.generate(
            global_search_meta_prompt.format(query=query, context=context),
            system_prompt
        )[0]

        output_dict = extract_json(output)
        return output_dict

    async def query(self, query: str):
        context = await self.search(query)
        truncated_context: str = self.truncation(str(context))

        output = self.client.generate(
            global_search_engine_prompt.format(query=query, context=truncated_context),
            system_prompt
        )[0]

        return self.extractor(output)