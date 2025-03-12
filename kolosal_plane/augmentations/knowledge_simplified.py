"""Simplified dataset augmentation for knowledge or data ingestion"""
from typing import Tuple, Dict

import polars as pl
import asyncio
from tqdm.asyncio import tqdm_asyncio

from kolosal_plane.augmentations.knowledge import Knowledge


class AsyncSimpleKnowledge(Knowledge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These will hold your dataset and metadata when ready
        self._augmented_data: pl.DataFrame = pl.DataFrame(
            schema={
                "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
                "document": pl.Utf8,
                "response": pl.Utf8
            }
        )
        self._metadata = {
            "llm_input_token_count": 0,
            "llm_output_token_count": 0,
            "tlm_input_token_count": 0,
            "tlm_output_token_count": 0
        }
        self._status = "Not started"
        self._task: asyncio.Task = None

    async def augmentate_async(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        self._status = "Running"
        # Step 1: Generate conversation starter questions for each document
        for document in self.documents:
            await asyncio.sleep(0)  # yield control
            built_knowledge_instructions = self.build_knowledge_instruction(
                document=document)
            self.conversation_starter_instruction = built_knowledge_instructions

            chat_histories, temp_llm_input, temp_llm_output = self.generate_conversation_starter()

            self._metadata["llm_input_token_count"] += temp_llm_input
            self._metadata["llm_output_token_count"] += temp_llm_output

            documents_data = [document] * len(chat_histories)
            new_df = pl.DataFrame({
                "chat_history": chat_histories,
                "document": documents_data,
                "response": [""] * len(chat_histories)
            })
            self._augmented_data = self._augmented_data.vstack(new_df)

        # Step 2: Loop for max_conversations rounds
        is_last = False
        for count in tqdm_asyncio(range(self.max_conversations), desc="Augmenting conversations"):
            await asyncio.sleep(0)
            if count == self.max_conversations - 1:
                is_last = True

            temporary_augmented_data = self._augmented_data.filter(
                pl.col("response") == "")
            self._augmented_data = self._augmented_data.filter(
                pl.col("response") != "")

            # Process data in batches
            batch_temporary_augmented_data = [
                temporary_augmented_data[i: i + self.batch_size]
                for i in range(0, len(temporary_augmented_data), self.batch_size)
            ]
            for batch in batch_temporary_augmented_data:
                try:
                    original_chat_histories = batch["chat_history"].to_list()
                    built_chat_histories = self.build_chat_histories(
                        documents=batch["document"].to_list(),
                        chat_histories=original_chat_histories
                    )

                    # Generate response using the appropriate model asynchronously if possible
                    if self.thinking_model:
                        responses, temp_tlm_input, temp_tlm_output = self.generate_response_thinking(
                            chat_histories=built_chat_histories
                        )
                        self._metadata["tlm_input_token_count"] += temp_tlm_input
                        self._metadata["tlm_output_token_count"] += temp_tlm_output
                    else:
                        responses, temp_llm_input, temp_llm_output = self.generate_response_llm(
                            chat_histories=built_chat_histories
                        )
                        self._metadata["llm_input_token_count"] += temp_llm_input
                        self._metadata["llm_output_token_count"] += temp_llm_output

                    # Save the response with original chat histories.
                    new_df = pl.DataFrame({
                        "chat_history": original_chat_histories,
                        "document": batch["document"],
                        "response": responses
                    })
                    self._augmented_data = self._augmented_data.vstack(new_df)
                except Exception as e:
                    print(f"Error in batch responses: {e}")
                    continue

                # Step 3: Generate follow-up questions if not the last round.
                if not is_last:
                    try:
                        questions, documents, temp_llm_input, temp_llm_output = self.generate_next_conversation(
                            llm=self.llm_model,
                            chat_histories=batch["chat_history"].to_list(),
                            responses=responses,
                            previous_documents=batch["document"].to_list()
                        )
                        self._metadata["llm_input_token_count"] += temp_llm_input
                        self._metadata["llm_output_token_count"] += temp_llm_output

                        generated_chat_histories = []
                        generated_chat_documents = []
                        for chat_history, response, question, document in zip(
                            batch["chat_history"].to_list(
                            ), responses, questions, documents
                        ):
                            new_chat_history = chat_history + [
                                {"role": "assistant", "content": response},
                                {"role": "user", "content": question}
                            ]
                            generated_chat_histories.append(new_chat_history)
                            generated_chat_documents.append(document)

                        new_df = pl.DataFrame({
                            "chat_history": generated_chat_histories,
                            "document": generated_chat_documents,
                            "response": [""] * len(generated_chat_histories)
                        })
                        self._augmented_data = self._augmented_data.vstack(
                            new_df)
                    except Exception as e:
                        print(f"Error in generating follow-up questions: {e}")

        self._status = "Finished"
        return self._augmented_data, self._metadata

    def start_augmentation(self):
        """
        Starts the augmentation process in the background.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.augmentate_async())
        return self._task

    def get_result(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Returns the current dataset and metadata.
        """
        return self._augmented_data, self._metadata

    def cancel_augmentation(self):
        """
        Cancels the running augmentation task.
        """
        if self._task and not self._task.done():
            self._task.cancel()

    def get_status(self) -> Tuple(str, Dict[int, int]):
        """
        Returns the current status of the augmentation process.
        """
        return self._status, self._metadata
