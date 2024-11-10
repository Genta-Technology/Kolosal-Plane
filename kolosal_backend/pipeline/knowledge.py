"""Component of the knowledge pipeline function"""

from typing import List, Dict

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct

from kolosal_backend.prompt_generation.knowledge_prompt import CONVERSATION_STARTER_PROMPT, CONVERSATION_SYSTEM_PROMPT


def build_knowledge_instruction(instruction: str, document: str) -> str:
    """
    Builds the instruction for generating knowledge-based conversations.
    Args:
        instruction (str): The instruction for generating knowledge-based conversations.
    Returns:
        str: The instruction for generating knowledge-based conversations.
    """
    return CONVERSATION_STARTER_PROMPT.format(instruction=instruction,
                                              document=document)


def build_knowledge_system(instruction: str, document: str) -> str:
    """
    Builds the instruction for generating knowledge-based conversations.
    Args:
        instruction (str): The instruction for generating knowledge-based conversations.
    Returns:
        str: The instruction for generating knowledge-based conversations.
    """
    return CONVERSATION_SYSTEM_PROMPT.format(instruction=instruction,
                                             document=document)


def build_chat_histories(instruction: str,
                         documents: List[str],
                         chat_histories: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
    """
    Builds chat histories by combining instructions, documents, and existing chat histories.
    Args:
        instruction (str): The instruction to be included in the chat history.
        documents (List[str]): A list of documents to be processed.
        chat_histories (List[List[Dict[str, str]]]): A list of existing chat histories, where each chat history is a list of dictionaries containing role and content.
    Returns:
        List[List[Dict[str, str]]]: A list of built chat histories, where each chat history is a list of dictionaries containing role and content.
    """
    built_chat_histories = []
    for document, chat_history in zip(documents, chat_histories):
        # Built the system prompt based on the document and instruction
        built_system = [{"role": "system", "content": build_knowledge_system(
            instruction=instruction, document=document)}]
        
        # Insert the system prompt at the beginning of the chat history
        chat_history = built_system + chat_history
        
        # Append the built chat history to the list
        built_chat_histories.append(chat_history)

    return built_chat_histories

def generate_next_conversation(llm: AsyncLLM,
                               chat_histories: List[List[Dict[str, str]]],
                               responses: List[str],
                               documents: List[str],
                               document_bank: List[str]) -> List[str]:
    
    generator = SelfInstruct(
        llm=llm,
        num_instructions=1
    )
    generator.load()
    
    # Built the input data for the generator such that it either:
    # 1. Generate the next conversation based on the chat history and response and the given document
    # 2. Generate the next conversation based on the chat history and response and the document bank (different knowledge)
    
    