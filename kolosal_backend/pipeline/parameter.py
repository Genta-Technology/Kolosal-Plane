"""
Parameter consumed for each type of pipeline
"""
from pydantic import BaseModel

from distilabel.llms import OpenAILLM, AsyncLLM
from typing import List


class PersonalizationParameter(BaseModel):
    """
    PersonalizationParameter is a data model for storing parameters related to conversation personalization.
    Attributes:
        conversation_starter_instruction (str): Instruction for generating conversation starters.
        conversation_personalization_instruction (str): Instruction for personalizing conversations.
        conversation_starter_count (int): Number of conversation starters to generate. Default is 10.
        max_conversations (int): Maximum number of conversations to generate. Default is 10.
        llm_model (AsyncLLM): The large language model to use for generating conversations. Default is OpenAILLM with model "gpt-4o".
        slm_model (AsyncLLM): The small language model to use for generating conversations. Default is OpenAILLM with model "gpt-4o-mini".
    """

    conversation_starter_instruction: str
    conversation_personalization_instruction: str
    conversation_starter_count: int = 10
    max_conversations: int = 10
    llm_model: AsyncLLM = OpenAILLM(model="gpt-4o")
    slm_model: AsyncLLM = OpenAILLM(model="gpt-4o-mini")


class KnowledgeParameter(BaseModel):
    """
    KnowledgeParameter is a data model that defines the parameters required for generating knowledge-based conversations.
    Attributes:
        documents (List[str]): A list of document strings to be used as the knowledge base.
        conversation_starter_instruction (str): Instructions for generating conversation starters.
        conversation_personalization_instruction (str): Instructions for personalizing the conversation.
        conversation_starter_count (int): The number of conversation starters to generate. Default is 10.
        max_conversations (int): The maximum number of conversations to generate. Default is 10.
        llm_model (AsyncLLM): The large language model to be used for generating conversations. Default is OpenAILLM with model "gpt-4o".
        slm_model (AsyncLLM): The small language model to be used for generating conversations. Default is OpenAILLM with model "gpt-4o-mini".
    """
    documents: List[str]
    conversation_starter_instruction: str
    conversation_personalization_instruction: str
    conversation_starter_count: int = 10
    max_conversations: int = 10
    llm_model: AsyncLLM = OpenAILLM(model="gpt-4o")
    slm_model: AsyncLLM = OpenAILLM(model="gpt-4o-mini")
