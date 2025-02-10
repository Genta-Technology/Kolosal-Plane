"""Simple Model Evaluation"""
from typing import Optional, List, Dict, Tuple

import polars as pl

from distilabel.llms.base import AsyncLLM
from distilabel.steps.tasks import ChatGeneration, SelfInstruct

from kolosal_plane.evaluation.evaluation_prompt import TEST_QUESTION_PROMPT


class Evaluation():
    """Evaluation Class for Finetuning Benchmark"""
    large_language_model: Optional[AsyncLLM]
    base_language_model: Optional[AsyncLLM]
    finetuned_language_model: Optional[AsyncLLM]

    documents: Optional[List[str]]
    system_prompt: Optional[str]
    batch_size: Optional[int]
    test_size: Optional[int]

    # Generate dataframe
    benchmark_data = pl.DataFrame(schema={
        "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
        "document": pl.Utf8,
        "llm_response": pl.Utf8,
        "blm_response": pl.Utf8,
        "flm_response": pl.Utf8
    })

    def __init__(self,
                 large_language_model: Optional[AsyncLLM] = None,
                 base_language_model: Optional[AsyncLLM] = None,
                 finetuned_language_model: AsyncLLM = None,

                 documents: Optional[List[str]] = None,
                 system_prompt: Optional[str] = None,
                 batch_size: Optional[int] = 16,
                 test_size: Optional[int] = 8):

        self.large_language_model = large_language_model
        self.base_language_model = base_language_model
        self.finetuned_language_model = finetuned_language_model

        self.documents = documents
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.test_size = test_size

    def benchmark(self) -> pl.DataFrame:
        # Generate questions for each douments
        built_chat_history, documents = self.generate_test_questions()

        # Answer using all AI types
        llm_response = self.generate_response_llm(built_chat_history)
        blm_response = self.generate_response_blm(built_chat_history)
        flm_response = self.generate_response_flm(built_chat_history)

        # Scoring (Next Step)

        # Dataframe
        self.benchmark_data = self.benchmark_data.vstack(pl.DataFrame({
            "chat_history": built_chat_history,
            "document": documents,
            "llm_response": llm_response,
            "blm_response": blm_response,
            "flm_response": flm_response
        }))

    def generate_test_questions(self) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Generates test questions based on provided documents using a large language model.
        This method creates a series of test questions by processing each document through
        a self-instruction model. It builds both chat history and corresponding document lists.
        Returns:
            Tuple[List[Dict[str, str]], List[str]]: A tuple containing:
                - chat_history: List of conversation sequences, where each sequence is a list
                  of dictionaries with 'role' and 'content' keys
                - documents: List of corresponding documents for each conversation in chat_history
        Example structure of returned chat_history:
            [
                [
                    {"role": "system", "content": "system_prompt"},  # if system_prompt is provided
                    {"role": "user", "content": "question"}
                ],
                ...
            ]
        Attributes used:
            self.large_language_model: The LLM instance used for generation
            self.test_size: Integer determining the number of test questions to generate
            self.documents: List of documents to generate questions from
            self.system_prompt: Optional system prompt to include in chat history
        """

        # Built Chat History
        chat_history = []

        # Built Documents
        documents = []

        # Initialized the SelfInstruct Instance
        generator = SelfInstruct(llm=self.large_language_model,
                                 num_instructions=self.test_size)

        # Load necessary resources for the generator
        generator.load()

        for document in self.documents:
            # Built the knowledge instruction
            test_question_prompt = TEST_QUESTION_PROMPT.format(
                document=document)

            # Process the input and generate conversation starters
            result = next(generator.process(
                [{"input": test_question_prompt}]))

            # Extract the 'instructions' from the result
            conversation_starters = result[0].get("instructions", [])
            # Transform the list of strings into chat history format, ignore system prompt if None
            chat_history += [[{"role": "user", "content": starter}] if self.system_prompt is None
                             else [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": starter}]
                             for starter in conversation_starters[:min(len(conversation_starters), self.test_size)]]

            documents += [document] * \
                min(len(conversation_starters), self.test_size)

        return chat_history, documents

    def generate_response(
        self,
        language_model: AsyncLLM,
        chat_histories: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generates responses for a list of chat histories using a language model.

        Args:
            lm (AsyncLLM): The asynchronous language model to use for generating responses.
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history 
                is a list of dictionaries containing messages. Each dictionary should have keys 'role' 
                and 'content' representing the role of the speaker and the message content respectively.
            **kwargs: Additional keyword arguments to be passed to the ChatGeneration class.

        Returns:
            List[str]: A list of generated responses corresponding to each chat history.
        """

        # Initialize the chat generator with the given language model
        generator = ChatGeneration(llm=language_model, **kwargs)

        # Load the generator
        generator.load()

        all_responses = []

        # Process each chat history individually
        for chat_history in chat_histories:
            try:
                # Generate a response for the current chat history
                result = next(generator.process([{"messages": chat_history}]))

                # Extract the 'generation' field from the response
                response = result[0]["generation"]

                # Accumulate the response
                all_responses.append(response)

            except (RuntimeError, ValueError, TypeError) as e:
                # Print the error message for the failed generation
                print(f"Error processing chat history: {str(e)}")
                # Add a default response for the failed chat history
                all_responses.append("FAILED RESPONSE")

        return all_responses

    def generate_response_llm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generate responses using a large language model based on chat histories.
        Args:
            chat_histories (List[List[Dict[str, str]]]): List of chat histories, where each chat history is a list of 
                dictionaries containing conversation messages and metadata.
            **kwargs: Additional keyword arguments to pass to the underlying generate_response method.
        Returns:
            List[str]: List of generated responses from the language model corresponding to each chat history.
        """

        return self.generate_response(language_model=self.large_language_model, chat_histories=chat_histories, kwargs=kwargs)

    def generate_response_blm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generate responses using the base language model.
        This method processes chat histories through the base language model to generate responses.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat history lists, where each chat history
                contains dictionaries with chat messages and their metadata.
            **kwargs: Additional keyword arguments to pass to the generate_response method.
        Returns:
            List[str]: A list of generated responses corresponding to each chat history.
        """

        return self.generate_response(language_model=self.base_language_model, chat_histories=chat_histories, kwargs=kwargs)

    def generate_response_flm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generates responses using a fine-tuned language model based on the provided chat histories.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat history lists, where each chat history
                is a list of dictionaries containing message details.
            **kwargs: Additional keyword arguments to pass to the generate_response method.
        Returns:
            List[str]: A list of generated responses from the fine-tuned language model corresponding
                to each chat history.
        """

        return self.generate_response(language_model=self.finetuned_language_model, chat_histories=chat_histories, kwargs=kwargs)
