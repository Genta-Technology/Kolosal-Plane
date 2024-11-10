"""Component of the knowledge pipeline function"""
import random
from typing import List, Dict

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct

from kolosal_backend.pipeline.pipeline_components import build_chat_history
from kolosal_backend.prompt_generation.knowledge_prompt import CONVERSATION_STARTER_PROMPT, CONVERSATION_SYSTEM_PROMPT, NEXT_QUESTION_SAME_TOPIC_PROMPT, NEXT_QUESTION_DIFFERENT_TOPIC_PROMPT


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
    """
    Generate the next conversation based on chat history, responses, and documents.
    This function uses a language model to generate the next conversation prompts
    based on the provided chat histories, responses, and documents. It randomly
    decides whether to use the given document or a document from the document bank
    for generating the next conversation.
    Args:
        llm (AsyncLLM): The language model to be used for generating the conversation.
        chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of dictionaries containing messages.
        responses (List[str]): A list of responses corresponding to each chat history.
        documents (List[str]): A list of documents corresponding to each chat history.
        document_bank (List[str]): A list of additional documents that can be used for generating the conversation.
    Returns:
        List[str]: A list of generated next conversation prompts.
        List[str]: A list of documents used for generating the conversation.
    """
    generator = SelfInstruct(
        llm=llm,
        num_instructions=1
    )
    generator.load()

    # Built the input data for the generator such that it either:
    # 1. Generate the next conversation based on the chat history and response and the given document
    # 2. Generate the next conversation based on the chat history and response and the document bank (different knowledge)

    input_data = []
    documents_used = []

    for chat_history, response, document in zip(chat_histories, responses, documents):
        built_chat_history = build_chat_history(chat_history)
        if random.choice([True, False]):
            # Option A: Generate the next conversation based on the chat history, response, and the given document
            input_data.append({"input": NEXT_QUESTION_SAME_TOPIC_PROMPT.format(chat_history=built_chat_history,
                                                                               response=response,
                                                                               document=document)})
            # Append the document to the used documents for further processing
            documents_used.append(document)
        else:
            # Option B: Generate the next conversation based on the chat history, response, and the document bank
            input_data.append({"input": NEXT_QUESTION_DIFFERENT_TOPIC_PROMPT.format(chat_history=built_chat_history,
                                                                                    response=response,
                                                                                    document=random.choice(document_bank))})
            # Append the document to the used documents for further processing
            documents_used.append(document)

    result = next(generator.process(input_data))
    next_questions = [next_question["instructions"][0]
                      for next_question in result]
    return next_questions, documents_used
