from typing import Iterator, List


def generate_prompt_batches(
    system_prompt: str,
    input_texts: List[str],
    batch_size: int,
    template: str = "{system}\n{input}"
) -> Iterator[List[str]]:
    """
    Generate batches of prompts by combining a system prompt with individual input texts.

    :param system_prompt: A single system prompt that will be combined with each input text.
    :param input_texts: A list of input texts to be processed in batches.
    :param batch_size: The number of input texts to include in each batch.
    :param template: A template string that defines how the system prompt and input text are combined.
                     Defaults to "{system}\n{input}".

    :return: An iterator that yields lists of formatted prompts, where each list represents a batch.

    Example:
        >>> system_prompt = "You are a helpful assistant."
        >>> input_texts = ["What is AI?", "Explain quantum computing."]
        >>> for batch in generate_prompt_batches(system_prompt, input_texts, batch_size=1):
        ...     print(batch)
        ['You are a helpful assistant.\nWhat is AI?']
        ['You are a helpful assistant.\nExplain quantum computing.']
    """
    for i in range(0, len(input_texts), batch_size):
        batch_inputs = input_texts[i:i + batch_size]
        batch_prompts = [
            template.format(system=system_prompt, input=input_text)
            for input_text in batch_inputs
        ]

        yield batch_prompts