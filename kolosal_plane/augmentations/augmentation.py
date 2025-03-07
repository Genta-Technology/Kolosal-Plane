"""Augmentation Class"""
from typing import Optional, List, Dict

from distilabel.models.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct, ChatGeneration, QualityScorer

from kolosal_plane.utils.llm import get_llm, get_slm


class Augmentation():
    conversation_personalization_instruction: str
    conversation_starter_instruction: str
    system_prompt: Optional[str]
    documents: Optional[List[str]]
    conversation_starter_count: Optional[int]
    max_conversations: Optional[int]
    batch_size: Optional[int]
    llm_model: Optional[AsyncLLM]
    slm_model: Optional[AsyncLLM]
    thinking_model: Optional[AsyncLLM]

    def __init__(self,
                 conversation_starter_instruction: str,
                 conversation_personalization_instruction: str,
                 system_prompt: Optional[str] = None,
                 # Only to be used for knowledge augmentation
                 documents: Optional[List[str]] = None,
                 conversation_starter_count: Optional[int] = 10,
                 max_conversations: Optional[int] = 10,
                 # Limit the request per second
                 batch_size: Optional[int] = 16,
                 llm_model: Optional[AsyncLLM] = None,
                 slm_model: Optional[AsyncLLM] = None,
                 thinking_model: Optional[AsyncLLM] = None):

        self.conversation_starter_instruction = conversation_starter_instruction
        self.conversation_personalization_instruction = conversation_personalization_instruction
        self.system_prompt = system_prompt
        self.documents = documents
        self.conversation_starter_count = conversation_starter_count
        self.max_conversations = max_conversations
        self.batch_size = batch_size
        self.llm_model = llm_model
        self.slm_model = slm_model
        self.thinking_model = thinking_model

        # Automatically load LLM and SLM models if not provided
        if self.llm_model is None:
            self.llm_model = get_llm()
        if self.slm_model is None:
            self.slm_model = get_slm()

    def generate_conversation_starter(self, **kwargs) -> List[Dict[str, str]]:
        """
        Generate conversation starters using the SelfInstruct instance.
        This method initializes a SelfInstruct instance with the provided language model and 
        the number of instructions to generate. It then processes the input instruction to 
        generate conversation starters and formats them into a chat history format.
        Args:
            **kwargs: Additional keyword arguments to pass to the SelfInstruct instance.
        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the chat history, 
            where each dictionary contains the role ('user' or 'system') and the content 
            of the conversation starter.
        Raises:
            ValueError: If the result does not contain valid instructions.
            RuntimeError: If an error occurs while generating conversation starters.
        """

        try:
            # Initialize the SelfInstruct instance
            generator = SelfInstruct(
                llm=self.llm_model,
                num_instructions=self.conversation_starter_count,
                **kwargs
            )

            # Load necessary resources for the generator
            generator.load()

            # Process the input and generate conversation starters
            result = next(generator.process(
                [{"input": self.conversation_starter_instruction}]))

            # Extract the 'instructions' from the result
            conversation_starters = result[0].get("instructions", [])

            # Ensure the output is a list of instructions
            if not isinstance(conversation_starters, list):
                raise ValueError(
                    "The result does not contain valid instructions.")

            # Transform the list of strings into chat history format, ignore system prompt if None
            chat_history = [[{"role": "user", "content": starter}] if self.system_prompt is None
                            else [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": starter}]
                            for starter in conversation_starters[:min(len(conversation_starters), self.conversation_starter_count)]]

            return chat_history

        except (RuntimeError, ValueError, TypeError) as e:
            raise RuntimeError(
                f"An error occurred while generating conversation starters: {str(e)}") from e

    def generate_response_llm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generates responses using a language model in batches.

        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is 
                a list of dictionaries containing the chat messages.
            **kwargs: Additional keyword arguments to be passed to the response generation function.

        Returns:
            List[str]: A list of generated responses.
        """

        return self.generate_response(self.llm_model, chat_histories, **kwargs)
    
    def generate_response_thinking(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generates response thinking using a thinking model.
        This method processes chat histories through the thinking model to generate responses.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat history sequences, where each sequence
                contains dictionaries with message content and metadata.
            **kwargs: Additional keyword arguments to pass to the generate_response method.
        Returns:
            List[str]: A list of generated responses from the thinking model.
        See Also:
            generate_response: The base method used for generating responses.
        """
        
        return self.generate_response(self.thinking_model, chat_histories, **kwargs)

    def generate_response_slm(self, chat_histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generates responses using a language model.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of dictionaries containing the chat messages.
            **kwargs: Additional keyword arguments to be passed to the response generation function.
        Returns:
            List[str]: A list of generated responses.
        """
        return self.generate_response(self.slm_model, chat_histories, **kwargs)

    def generate_response(
        self,
        lm: AsyncLLM,
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
        generator = ChatGeneration(llm=lm, **kwargs)

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

    def comparison_score(self,
                         chat_histories: List[List[Dict[str, str]]],
                         llm_responses: List[str],
                         slm_responses: List[str],
                         **kwargs
                         ) -> List[int]:
        """
        Asynchronously computes comparison scores between LLM and SLM responses based on chat histories.
        Args:
            chat_histories (List[List[Dict[str, str]]]): A list of chat histories, where each chat history is a list of dictionaries containing chat messages.
            llm_responses (List[str]): A list of responses generated by the large language model.
            slm_responses (List[str]): A list of responses generated by the small language model.
        Returns:
            List[int]: A list of integers where 0 indicates the LLM response is better, 1 indicates the SLM response is better, and 0 is the default in case of an error.
        """

        # Initialize and load the QualityScorer
        generator = QualityScorer(llm=self.llm_model, **kwargs)
        generator.load()

        # Prepare the input data for scoring
        input_data = [
            {
                "instruction": chat_history,
                "responses": [llm_response, slm_response]
            }
            for chat_history, llm_response, slm_response
            in zip(chat_histories, llm_responses, slm_responses)
        ]

        all_scores_list = generator.process(input_data)

        # Now compute the comparison results from the combined scores
        result_scores = []
        for score_dict in all_scores_list:
            try:
                # Retrieve scores, defaulting to [0, 0] if not present
                scores = score_dict.get("scores", [0, 0])
                # Replace any None values with 0
                scores = [s if s is not None else 0 for s in scores]
                # Find the index of the highest score
                max_score = max(scores)
                max_index = scores.index(max_score)
                result_scores.append(max_index)
            except (KeyError, ValueError, TypeError) as e:
                # Print error for debugging purposes
                print(f"Error processing score dict {score_dict}: {str(e)}")
                # Default to 0 if an error occurs
                result_scores.append(0)

        return result_scores

    def convert_chat_history(self, chat_history: List[Dict]) -> str:
        """
        Builds a chat history string from a list of chat entries.
        Args:
            chat_history (List[Dict]): A list of dictionaries representing chat entries. 
                                    Each dictionary should have 'role' and 'content' keys.
        Returns:
            str: A string representation of the chat history, with each entry formatted as "role: content".
        """

        built_chat = ""
        for entry in chat_history:
            built_chat += f"{entry['role']}: {entry['content']}\n"
        return built_chat
