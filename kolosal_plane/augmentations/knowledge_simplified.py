"""Simplified dataset augmentation for knowledge or data ingestion"""
import polars as pl
from tqdm import tqdm

from kolosal_plane.augmentations.knowledge import Knowledge


class SimpleKnowledge(Knowledge):
    def augmentate(self) -> pl.DataFrame:
        """
        Augments the dataset using a simplified approach:
          - Generate conversation starter questions from documents.
          - In each conversation round, generate an LLM response (using a built chat history)
            but save the response paired with the original user chat history (without the extra context).
          - Optionally generate a follow-up question (if not the last round) and add new rows.
        Returns:
            pl.DataFrame: A DataFrame with columns 'chat_history', 'document', and 'response'.
        """
        augmented_data = pl.DataFrame(schema={
            "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
            "document": pl.Utf8,
            "response": pl.Utf8
        })

        # Step 1: Generate conversation starter questions for each document
        for document in self.documents:
            # Build knowledge instructions for this document.
            built_knowledge_instructions = self.build_knowledge_instruction(
                document=document)
            self.conversation_starter_instruction = built_knowledge_instructions

            # Generate conversation starter chat histories
            chat_histories = self.generate_conversation_starter()
            documents_data = [document] * len(chat_histories)

            augmented_data = augmented_data.vstack(pl.DataFrame({
                "chat_history": chat_histories,
                "document": documents_data,
                "response": [""] * len(chat_histories)
            }))

        # Loop for max_conversations rounds to build and expand the dataset.
        is_last = False
        for count in tqdm(range(self.max_conversations), desc="Augmenting conversations"):
            if count == self.max_conversations - 1:
                is_last = True

            # Get rows that have not been answered yet.
            temporary_augmented_data = augmented_data.filter(
                pl.col("response") == "")

            # Remove the unanswered rows from augmented_data so that we do not process them twice.
            augmented_data = augmented_data.filter(pl.col("response") != "")

            # Batch the data to catch and isolate errors
            batch_temporary_augmented_data = [
                temporary_augmented_data[i: i + self.batch_size]
                for i in range(0, len(temporary_augmented_data), self.batch_size)
            ]

            for batch in batch_temporary_augmented_data:
                batch_status = True
                try:
                    # Keep the original chat histories for saving.
                    original_chat_histories = batch["chat_history"].to_list()

                    # Build chat histories for the generation call (adds extra context for the LLM).
                    built_chat_histories = self.build_chat_histories(
                        documents=batch["document"].to_list(),
                        chat_histories=original_chat_histories
                    )

                    responses = []

                    # Generate LLM or thinking responses using the built chat histories.
                    if self.thinking_model:
                        responses = self.generate_response_thinking(
                            chat_histories=built_chat_histories)
                    else:
                        responses = self.generate_response_llm(
                            chat_histories=built_chat_histories)

                    # Save the response along with the original chat histories.
                    augmented_data = augmented_data.vstack(pl.DataFrame({
                        "chat_history": original_chat_histories,
                        "document": batch["document"],
                        "response": responses
                    }))
                except Exception as e:
                    print(
                        f"Error in handling batch responses, omitting this batch: {e}")
                    batch_status = False

                # Step 3: (If not the last round) Generate follow-up questions and expand the dataset.
                if batch_status and not is_last:
                    try:
                        questions, documents = self.generate_next_conversation(
                            llm=self.llm_model,
                            chat_histories=batch["chat_history"].to_list(),
                            responses=responses,
                            previous_documents=batch["document"].to_list()
                        )

                        generated_chat_histories = []
                        generated_chat_documents = []
                        for chat_history, response, question, document in zip(
                            batch["chat_history"].to_list(),
                            responses,
                            questions,
                            documents
                        ):
                            # Append the new conversation to the original chat history.
                            new_chat_history = chat_history + [
                                {"role": "assistant", "content": response},
                                {"role": "user", "content": question}
                            ]
                            generated_chat_histories.append(new_chat_history)
                            generated_chat_documents.append(document)

                        # Save the new rows with an empty response to be processed in a later round.
                        augmented_data = augmented_data.vstack(pl.DataFrame({
                            "chat_history": generated_chat_histories,
                            "document": generated_chat_documents,
                            "response": [""] * len(generated_chat_histories)
                        }))
                    except Exception as e:
                        print(
                            f"Error in handling batch followup questions: {e}")

        return augmented_data
