"""Selection of SLM and LLM for the API"""
import os
from distilabel.llms import AzureOpenAILLM, AsyncLLM, OpenAILLM

# Modify the code as necessary according to which llm and slm you are using
llm = AzureOpenAILLM(base_url=os.getenv("LLM_ENDPOINT"),
                     api_key=os.getenv("LLM_API_KEY"),
                     api_version="2024-02-15-preview",
                     model=os.getenv("LLM_MODEL", "gpt-4o"),
                     generation_kwargs={
                         "max_new_tokens": 1024
})

slm = AzureOpenAILLM(base_url=os.getenv("SLM_ENDPOINT"),
                     api_key=os.getenv("SLM_API_KEY"),
                     api_version="2024-02-15-preview",
                     model=os.getenv("SLM_MODEL", "gpt-4o-mini"),
                     generation_kwargs={
                         "max_new_tokens": 1024
})

# Using Genta API as the SLM provider (Uncomment this code if you want to use Genta API)
# slm = OpenAILLM(base_url=os.getenv("SLM_ENDPOINT"),
#                 api_key=os.getenv("SLM_API_KEY"),
#                 model=os.getenv("SLM_MODEL", "Llama-3.2-1B-Instruct"),
#                 generation_kwargs={
#                     "max_new_tokens": 1024
# })


def get_llm() -> AsyncLLM:
    """
    Retrieve the instance of the asynchronous language model (LLM).
    Returns:
        AsyncLLM: The instance of the asynchronous language model.
    """

    return llm


def get_slm() -> AsyncLLM:
    """
    Retrieve the SLM (Service Level Management) instance.
    Returns:
        AsyncLLM: The SLM instance.
    """

    return slm
