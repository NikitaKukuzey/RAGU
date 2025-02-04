from pydantic_settings import BaseSettings


class Registrable:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            if cls not in cls._registry:
                cls._registry[cls] = {}
            registry = cls._registry[cls]
            registry[name] = subclass
            return subclass
        return wrapper

    @classmethod
    def get_by_name(cls, class_name: str, *args, **kwargs):
        print(kwargs)
        registry = cls._registry.get(cls, {})
        if class_name not in registry:
            raise ValueError(f"Class named '{class_name}' not found.")
        return registry[class_name](*args, kwargs)
    

class Settings(BaseSettings):
    llm_base_url: str
    llm_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()
