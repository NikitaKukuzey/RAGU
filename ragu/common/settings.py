import os
import openai
import logging
import pandas as pd
from datetime import datetime
from pydantic_settings import BaseSettings


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
    llm_model_name: str
    llm_base_url: str
    llm_api_key: str

    class Config:
        """
        Configuration class to specify the location of environment variables.

        :param env_file: The file where environment variables are loaded from (e.g., `.env`).
        """
        env_file = ".env"
        
    @classmethod
    def from_env(cls, env_path: str | None):
        """
        Creates an instance of `Settings`, loading environment variables from a specified file.

        :param env_path: Path to the `.env` file (optional). If None, uses the default `.env`.
        :return: An instance of `Settings` with values loaded from the specified file.
        """
        return cls(_env_file=env_path) if env_path else cls()


# Logging
openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

current_dir = os.getcwd()
temp_dir = os.path.join(current_dir, "temp")
logs_dir = os.path.join(temp_dir, "logs")
outputs_dir = os.path.join(temp_dir, "outputs")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run_output_dir = os.path.join(outputs_dir, current_time)
os.makedirs(run_output_dir, exist_ok=True)

log_filename = os.path.join(logs_dir, f"ragu_logs_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    filename=log_filename,
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_outputs(df: pd.DataFrame, filename: str):
    """
    Save DataFrame in specified directory.

    :param df: DataFrame to save.
    :param filename: filename for the saved file.
    """
    filepath = os.path.join(run_output_dir, f"{filename}.parquet")
    df.to_parquet(filepath, index=False)
    logging.info(f"Outputs saved in: {filepath}")


# Instantiate the settings object and load values from the environment
settings = Settings()

