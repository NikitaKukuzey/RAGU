import re

DEFAULT_DELIMITER = "<|>"


def build_prompt(template: str, delimiter: str = DEFAULT_DELIMITER, **kwargs):
    """
    Build a prompt with the given arguments.
    """
    kwargs["delimiter"] = delimiter
    def replacer(match):
        key = match.group(1)
        value = kwargs.get(key, [match.group(0)])
        return ", ".join(value) if isinstance(value, list) else str(value)

    return re.sub(r"\{(\w+)]", replacer, template)
