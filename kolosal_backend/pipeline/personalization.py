"""Component of the personalization pipeline function"""
from typing import List, Dict

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct

from kolosal_backend.pipeline.pipeline_components import build_chat_history
from kolosal_backend.prompt_generation.personalization_prompt import NEXT_QUESTION_PROMPT


def generate_next_conversation(llm: AsyncLLM,
                               chat_histories: List[Dict[str, str]],
                               responses: str,
                               **kwargs) -> List[Dict[str, str]]:
    """
    Asynchronously generates the next conversation prompts based on chat histories and responses.
    Args:
        llm (AsyncLLM): The language model to use for generating the next conversation.
        chat_histories (List[Dict[str, str]]): A list of dictionaries containing the chat histories.
        responses (str): The responses to the chat histories.
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the next conversation prompts.
    """

    generator = SelfInstruct(
        llm=llm,
        num_instructions=1,
        **kwargs
    )
    generator.load()

    input_data = [{"input": NEXT_QUESTION_PROMPT.format(chat_history=build_chat_history(chat_history),
                                                        response=response)}
                  for chat_history, response in zip(chat_histories, responses)]

    result = next(generator.process(input_data))
    next_questions = [next_question["instructions"][0]
                      for next_question in result]
    return next_questions
