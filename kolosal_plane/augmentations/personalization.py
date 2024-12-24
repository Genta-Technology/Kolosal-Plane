"""Dataset augmentation for personalization."""
import copy
from typing import List, Dict

import polars as pl
from tqdm import tqdm

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct

from kolosal_plane.augmentations.augmentation import Augmentation
from kolosal_plane.augmentations.prompt.personalization_prompt import NEXT_QUESTION_PROMPT


class Personalization(Augmentation):
    def augmentate(self):
        """
        Given an instruction created based on user interaction preference,
        this function generate an augmented dataset of conversation to finetune a SLM to fit the user interaction preference
        """
        augmented_data = pl.DataFrame(schema={
            "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
            "slm_response": pl.Utf8,
            "llm_response": pl.Utf8,
            "scores": pl.Int64
        })

        # Step 1: Generate conversation starter data
        temporary_augmented_data = pl.DataFrame({
            "chat_history": self.generate_conversation_starter()
        })

        # Loop the conversation n times to generate augmented data
        for _ in tqdm(range(self.max_conversations)):
            # Step 2 Generate SLM and LLM response
            slm_responses = self.generate_response_slm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                input_batch_size=10)
            llm_responses = self.generate_response_llm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                input_batch_size=10)

            temporary_augmented_data = temporary_augmented_data.with_columns(
                pl.Series("slm_response", slm_responses),
                pl.Series("llm_response", llm_responses)
            )

            # Step 3 Generate responses score
            scores = self.comparison_score(chat_histories=temporary_augmented_data["chat_history"].to_list(),
                                           llm_responses=llm_responses,
                                           slm_responses=slm_responses,
                                           input_batch_size=10)

            temporary_augmented_data = temporary_augmented_data.with_columns(
                pl.Series("scores", scores)
            )

            # Step 4 Generate a followup question based on the chat history
            slm_questions = self.generate_next_conversation_slm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                responses=slm_responses)
            llm_questions = self.generate_next_conversation_llm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                responses=llm_responses)

            # Step 5 Geneare a new chat history dataset based on the questions asked
            generated_chat_histories = []
            for chat_history, slm_response, llm_response, slm_question, llm_question in zip(temporary_augmented_data["chat_history"].to_list(), slm_responses, llm_responses, slm_questions, llm_questions):
                # Append for generated dataset based on slm
                generated_chat_histories.append(
                    chat_history + [{"role": "assistant", "content": slm_response},
                                    {"role": "user", "content": slm_question}]
                )

                # Append for generated dataset based on llm
                generated_chat_histories.append(
                    chat_history + [{"role": "assistant", "content": llm_response},
                                    {"role": "user", "content": llm_question}]
                )

            # Save the augmented data
            augmented_data = augmented_data.vstack(
                copy.deepcopy(temporary_augmented_data))
            temporary_augmented_data = pl.DataFrame({
                "chat_history": generated_chat_histories
            })

        return augmented_data

    def generate_next_conversation_llm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, str]]:
        """
        Generates responses for a list of chat histories using a language model.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of 
                dictionaries containing messages. Each dictionary should have keys 'role' and 'content' representing 
                the role of the speaker and the message content respectively.
            **kwargs: Additional keyword arguments to be passed to the ChatGeneration class.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the next conversation prompts.
        """

        return self.generate_response(self.llm_model, chat_histories, **kwargs)

    def generate_next_conversation_slm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, str]]:
        """
        Generates responses for a list of chat histories using a language model.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of 
                dictionaries containing messages. Each dictionary should have keys 'role' and 'content' representing 
                the role of the speaker and the message content respectively.
            **kwargs: Additional keyword arguments to be passed to the ChatGeneration class.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the next conversation prompts.
        """

        return self.generate_response(self.slm_model, chat_histories, **kwargs)

    def generate_next_conversation(self,
                                   llm: AsyncLLM,
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

        input_data = [{"input": NEXT_QUESTION_PROMPT.format(chat_history=self.convert_chat_history(chat_history),
                                                            response=response)}
                      for chat_history, response in zip(chat_histories, responses)]

        result = next(generator.process(input_data))
        next_questions = [next_question["instructions"][0]
                          for next_question in result]
        return next_questions
