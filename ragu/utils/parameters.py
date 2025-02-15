import yaml

from ragu.common.parameters import (
    ChunkerParameters,
    TripletExtractorParameters,
    RerankerParameters,
    GeneratorParameters
)

PARAMETER_CLASSES = {
    "chunker": ChunkerParameters,
    "triplet": TripletExtractorParameters,
    "reranker": RerankerParameters,
    "generator": GeneratorParameters,
}


def get_parameters(path_to_yaml: str):
    with open(path_to_yaml, 'r') as f:
        parameters = yaml.safe_load(f)

    parsed_params = {}
    for key, param_cls in PARAMETER_CLASSES.items():
        section = parameters.get(key)
        if section is None:
            raise ValueError(f"No '{key}' section in config.")
        try:
            parsed_params[key] = param_cls(**section)
        except TypeError as e:
            raise ValueError(
                f"Error parsing {param_cls.__name__} with section {section}: {e}"
            )

    return (
        parsed_params["chunker"],
        parsed_params["triplet"],
        parsed_params["reranker"],
        parsed_params["generator"],
    )
