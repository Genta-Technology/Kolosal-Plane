"""Simplified dataset augmentation for knowledge or data ingestion"""
import copy

import polars as pl
from tqdm import tqdm

from kolosal_plane.augmentations.knowledge import Knowledge


class SimpleKnowledge(Knowledge):
    def augmentate(self):
        augmented_data = pl.DataFrame(schema={
            "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
            "document": pl.Utf8,
            "response": pl.Utf8
        })

        # Step 1: Generate conversation starter question based on the given documents
        for document in self.documents:
            built_knowledge_instructions = self.build_knowledge_instruction(
                document=document)
            self.conversation_starter_instruction = built_knowledge_instructions

            chat_histories = self.generate_conversation_starter()

            documents_data = [document] * len(chat_histories)

            augmented_data = augmented_data.vstack(pl.DataFrame({
                "chat_history": chat_histories,
                "document": documents_data,
                "response": [""] * len(chat_histories)
            }))

        # Loop the conversation according to the instruction max conversation times to generate augmented data
        for _ in tqdm(range(self.max_conversations)):
            # Find not answered dataset
            temporary_augmented_data = augmented_data.filter(
                pl.col("response") == "")

            # Remove the non answered dataset from the augmented_data
            augmented_data = augmented_data.filter(
                pl.col("response") != ""
            )

            # Batching for catchinge errors
            batch_temporary_augmented_data = [temporary_augmented_data[i: i+self.batch_size]
                                              for i in range(0, len(temporary_augmented_data), self.batch_size)]

            for batch in batch_temporary_augmented_data:
                batch_status = True
                try:
                    # Step 2 Generate LLM response
                    built_chat_histories = self.build_chat_histories(
                        documents=batch["document"].to_list(
                        ),
                        chat_histories=batch["chat_history"].to_list())

                    responses = self.generate_response_llm(
                        chat_histories=built_chat_histories)

                    # Save the response to the built dataset
                    augmented_data.vstack(pl.DataFrame({
                        "chat_history": batch["chat_history"],
                        "document": batch["document"],
                        "response": responses
                    }))
                except Exception as e:
                    print(
                        f"Error in handling batch responses, omitting the batch from the main dataset: {e}")
                    batch_status = False

                # Step 3 Generate a followup question based on the chat history and Document
                if batch_status:
                    try:
                        questions, documents = self.generate_next_conversation(
                            llm=self.llm_model,
                            chat_histories=batch["chat_history"].to_list(
                            ),
                            responses=responses,
                            previous_documents=batch["document"].to_list(
                            ))

                        # Step 4: Generate a new chat history dataset based on the questions asked
                        generated_chat_histories = []
                        generated_chat_documents = []
                        for chat_history, response, question, document in zip(
                            batch["chat_history"].to_list(),
                            responses,
                            questions,
                            documents
                        ):
                            # Append the new conversation to the chat history
                            new_chat_history = chat_history + [
                                {"role": "assistant", "content": response},
                                {"role": "user", "content": question}
                            ]

                        generated_chat_histories.append(new_chat_history)
                        generated_chat_documents.append(document)

                        # Save the new question and documents in the main dataset
                        augmented_data = augmented_data.vstack(pl.DataFrame({
                            "chat_history": generated_chat_histories,
                            "document": generated_chat_documents,
                            "response": [""] * len(generated_chat_histories)
                        }))
                    except Exception as e:
                        print(
                            f"Error in handling batch followup questions: {e}")

        return augmented_data
