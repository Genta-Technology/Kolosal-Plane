"""Selection of SLM and LLM for the API"""
import os
import json
from distilabel.llms import AzureOpenAILLM, AsyncLLM, OpenAILLM, AnthropicLLM

# Load configuration from config file
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)


def get_lm_instance(model_config):
    """
    Returns an instance of a language model based on the provided configuration.
    Args:
        model_config (dict): A dictionary containing the configuration for the model.
            - model_provider (str): The provider of the model (e.g., "azure", "openai", "genta", "fireworks", "anthropic").
            - model_name (str): The name of the model to be used.
            - model_parameters (dict, optional): Additional parameters for model generation.
    Returns:
        An instance of the specified language model.
    Raises:
        ValueError: If the model provider specified in the configuration is unsupported.
    """

    model_provider = model_config["model_provider"]
    model_name = model_config["model_name"]
    model_parameters = model_config.get("model_parameters", {})

    if model_provider == "azure":
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        return AzureOpenAILLM(base_url=azure_openai_endpoint,
                              api_key=azure_openai_api_key,
                              api_version=azure_api_version,
                              model=model_name,
                              generation_kwargs={"max_new_tokens": 1024, **model_parameters})
    elif model_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return OpenAILLM(api_key=openai_api_key,
                         model=model_name,
                         generation_kwargs={"max_new_tokens": 1024, **model_parameters})
    elif model_provider == "genta":
        genta_api_key = os.getenv("GENTA_API_KEY")
        return OpenAILLM(api_key=genta_api_key,
                         base_url="https://api.genta.tech",
                         model=model_name,
                         generation_kwargs={"max_new_tokens": 1024, **model_parameters})
    elif model_provider == "fireworks":
        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        return OpenAILLM(api_key=fireworks_api_key,
                         base_url="https://api.fireworks.ai/inference/v1",
                         model=model_name,
                         generation_kwargs={"max_new_tokens": 1024, **model_parameters})
    elif model_provider == "anthropic":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicLLM(api_key=anthropic_api_key,
                            model=model_name,
                            generation_kwargs={"max_new_tokens": 1024, **model_parameters})
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


def get_llm() -> AsyncLLM:
    """
    Retrieve the instance of the asynchronous language model (LLM).
    Returns:
        AsyncLLM: The instance of the asynchronous language model.
    """
    return get_lm_instance(config["llm"])


def get_slm() -> AsyncLLM:
    """
    Retrieve the SLM (Service Level Management) instance.
    Returns:
        AsyncLLM: The SLM instance.
    """
    return get_lm_instance(config["slm"])
