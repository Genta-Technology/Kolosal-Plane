"""Dataset augmentation for knowledge or data ingestion"""
import copy
import random
from typing import List, Dict, Tuple

import polars as pl
from tqdm import tqdm

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct

from kolosal_plane.augmentations.augmentation import Augmentation
from kolosal_plane.augmentations.prompt.knowledge_prompt import CONVERSATION_STARTER_PROMPT, CONVERSATION_SYSTEM_PROMPT, NEXT_QUESTION_SAME_TOPIC_PROMPT, NEXT_QUESTION_DIFFERENT_TOPIC_PROMPT


class Knowledge(Augmentation):
    def augmentate(self):
        augmented_data = pl.DataFrame(schema={
            "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
            "document": pl.Utf8,
            "slm_response": pl.Utf8,
            "llm_response": pl.Utf8,
            "scores": pl.Int64
        })

        temporary_augmented_data = pl.DataFrame(schema={
            "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
            "document": pl.Utf8
        })

        # Step 1: Generate conversation starter question based on the given documents
        for document in self.documents:
            built_knowledge_instructions = self.build_knowledge_instruction()

            chat_histories_document = self.generate_conversation_starter(
                instruction=built_knowledge_instructions)

            document_data = [document] * len(chat_histories_document)

            temporary_augmented_data = temporary_augmented_data.vstack(pl.DataFrame({
                "chat_history": chat_histories_document,
                "document": document_data
            }))

        # Loop the conversation according to the instruction max conversation times to generate augmented data
        for _ in tqdm(range(self.max_conversations)):
            # Step 2 Generate SLM and LLM response
            built_chat_histories = self.build_chat_histories(
                documents=temporary_augmented_data["document"].to_list(),
                chat_histories=temporary_augmented_data["chat_history"].to_list())

            slm_responses = self.generate_response_slm(
                chat_histories=built_chat_histories)
            llm_responses = self.generate_response_llm(
                chat_histories=built_chat_histories)

            temporary_augmented_data = temporary_augmented_data.with_columns(
                pl.Series("slm_response", slm_responses),
                pl.Series("llm_response", llm_responses)
            )

            # Step 3 Generate responses score
            scores = self.comparison_score(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                llm_responses=llm_responses,
                slm_responses=slm_responses)

            temporary_augmented_data = temporary_augmented_data.with_columns(
                pl.Series("scores", scores)
            )

            # Step 4 Generate a followup question based on the chat history and Document
            slm_questions, slm_documents = self.generate_next_conversation_slm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                responses=slm_responses,
                previous_documents=temporary_augmented_data["document"].to_list(
                ))
            llm_questions, llm_documents = self.generate_next_conversation_llm(
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                responses=slm_responses,
                previous_documents=temporary_augmented_data["document"].to_list(
                ))
            # Step 5 Geneare a new chat history dataset based on the questions asked
            generated_chat_histories = []
            generated_chat_documents = []
            for chat_history, slm_response, llm_response, slm_question, llm_question, slm_document, llm_document in zip(temporary_augmented_data["chat_history"].to_list(),
                                                                                                                        slm_responses,
                                                                                                                        llm_responses,
                                                                                                                        slm_questions,
                                                                                                                        llm_questions,
                                                                                                                        slm_documents,
                                                                                                                        llm_documents):
                # Append for generated dataset based on slm
                generated_chat_histories.append(
                    chat_history + [{"role": "assistant", "content": slm_response},
                                    {"role": "user", "content": slm_question}]
                )

                generated_chat_documents.append(slm_document)

                # Append for generated dataset based on llm
                generated_chat_histories.append(
                    chat_history + [{"role": "assistant", "content": llm_response},
                                    {"role": "user", "content": llm_question}]
                )
                generated_chat_documents.append(llm_document)

            # Save the augmented data
            augmented_data = augmented_data.vstack(
                copy.deepcopy(temporary_augmented_data))
            temporary_augmented_data = pl.DataFrame({
                "chat_history": generated_chat_histories,
                "document": generated_chat_documents
            })

        return augmented_data

    def build_knowledge_instruction(self) -> str:
        """
        Builds the instruction for generating knowledge-based conversations.
        Args:
            instruction (str): The instruction for generating knowledge-based conversations.
        Returns:
            str: The instruction for generating knowledge-based conversations.
        """
        return CONVERSATION_STARTER_PROMPT.format(instruction=self.conversation_starter_instruction,
                                                  document=self.documents)

    def build_knowledge_system(self,
                               document: str) -> str:
        """
        Builds the instruction for generating knowledge-based conversations.
        Args:
            instruction (str): The instruction for generating knowledge-based conversations.
        Returns:
            str: The instruction for generating knowledge-based conversations.
        """
        return CONVERSATION_SYSTEM_PROMPT.format(instruction=self.conversation_personalization_instruction,
                                                 document=document)

    def build_chat_histories(self,
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
            built_system = [{"role": "system", "content": self.build_knowledge_system(
                document=document)}]

            # Insert the system prompt at the beginning of the chat history
            chat_history = built_system + chat_history

            # Append the built chat history to the list
            built_chat_histories.append(chat_history)

        return built_chat_histories

    def generate_next_conversation_llm(self,

                                       chat_histories: List[List[Dict[str, str]]],
                                       responses: List[str],
                                       previous_documents: List[str]) -> Tuple[str, List[str]]:
        """
        Generates the next conversation responses using a language model.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of dictionaries containing the conversation.
            responses (List[str]): A list of responses to be used in generating the next conversation.
            previous_documents (List[str]): A list of previous documents relevant to the conversation.
        Returns:
            List[str]: A list of generated next conversation prompts.
            List[str]: A list of documents used for generating the conversation.
        """

        return self.generate_next_conversation(self.llm_model,
                                               chat_histories,
                                               responses,
                                               previous_documents)

    def generate_next_conversation_slm(self,
                                       chat_histories: List[List[Dict[str, str]]],
                                       responses: List[str],
                                       previous_documents: List[str],) -> Tuple[str, List[str]]:
        """
        Generates the next conversation responses using a language model.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of dictionaries containing the conversation.
            responses (List[str]): A list of responses to be used in generating the next conversation.
            previous_documents (List[str]): A list of previous documents relevant to the conversation.
        Returns:
            List[str]: A list of generated next conversation prompts.
            List[str]: A list of documents used for generating the conversation.
        """
        return self.generate_next_conversation(self.slm_model,
                                               chat_histories,
                                               responses,
                                               previous_documents)

    def generate_next_conversation(self,
                                   llm: AsyncLLM,
                                   chat_histories: List[List[Dict[str, str]]],
                                   responses: List[str],
                                   previous_documents: List[str],) -> Tuple[str, List[str]]:
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

        for chat_history, response, document in zip(chat_histories, responses, previous_documents):
            built_chat_history = self.convert_chat_history(chat_history)
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
                                                                                        document=random.choice(self.documents))})
                # Append the document to the used documents for further processing
                documents_used.append(document)

        result = next(generator.process(input_data))
        next_questions = [next_question["instructions"][0]
                          for next_question in result]
        return next_questions, documents_used