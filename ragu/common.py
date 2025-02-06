from typing import ClassVar
from pydantic_settings import BaseSettings


class Registrable:
    """
    A base class for registering subclasses dynamically and retrieving them by name.

    This class manages a registry of subclasses that can be registered and retrieved using 
    their names. The `register` method registers a subclass, and the `get` method retrieves 
    a registered subclass.

    :param name: The name used to register the subclass.
    :param class_name: The name of the class to retrieve from the registry.
    :param *args: Arguments passed to the constructor of the registered class.
    :param **kwargs: Keyword arguments passed to the constructor of the registered class.
    :return: A subclass of the `Registrable` class, instantiated with the provided arguments.
    """
    _registry: ClassVar[dict] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a subclass with a given name in the class's registry.

        :param name: The name under which the subclass will be registered.
        :return: The decorator that registers the subclass.
        """
        def wrapper(subclass):
            if cls not in cls._registry:
                cls._registry[cls] = {}
            registry = cls._registry[cls]
            registry[name] = subclass
            return subclass
        return wrapper

    @classmethod
    def get(cls, class_name: str, *args, **kwargs):
        """
        Retrieves a registered subclass by name and instantiates it.

        If the class is not found in the registry, a ValueError is raised.

        :param class_name: The name of the class to retrieve from the registry.
        :param *args: Arguments passed to the constructor of the subclass.
        :param **kwargs: Keyword arguments passed to the constructor of the subclass.
        :return: An instance of the registered subclass.
        :raises ValueError: If the class with the given name is not found in the registry.
        """
        registry = cls._registry.get(cls, {})
        if class_name not in registry:
            raise ValueError(f"Class named '{class_name}' not found.")
        return registry[class_name](class_name, *args, **kwargs)
    

class Settings(BaseSettings):
    """
    A configuration class that loads settings from environment variables using Pydantic.

    This class defines the settings required to connect to an LLM API, such as the base URL 
    and API key. These values are loaded from environment variables, as specified in the 
    `.env` file.

    :param llm_base_url: The base URL for the LLM API.
    :param llm_api_key: The API key used for authenticating requests to the LLM API.
    :return: An instance of the `Settings` class with values loaded from the environment.
    """
    llm_base_url: str
    llm_api_key: str

    class Config:
        """
        Configuration class to specify the location of environment variables.

        :param env_file: The file where environment variables are loaded from (e.g., `.env`).
        """
        env_file = ".env"


import logging
import openai

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

# Instantiate the settings object and load values from the environment
settings = Settings()
