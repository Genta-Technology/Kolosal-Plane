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

            chat_histories_document = self.generate_conversation_starter()

            document_data = [document] * len(chat_histories_document)

            temporary_augmented_data = temporary_augmented_data.vstack(pl.DataFrame({
                "chat_history": chat_histories_document,
                "document": document_data
            }))

         # Loop the conversation according to the instruction max conversation times to generate augmented data
        for _ in tqdm(range(self.max_conversations)):
            # Step 2 Generate LLM response: Could contained failed batch
            built_chat_histories = self.build_chat_histories(
                documents=temporary_augmented_data["document"].to_list(),
                chat_histories=temporary_augmented_data["chat_history"].to_list())

            responses = self.generate_response_llm(
                chat_histories=built_chat_histories)

            # Step 3 Generate a followup question based on the chat history and Document
            questions, documents = self.generate_next_conversation(
                llm=self.llm_model,
                chat_histories=temporary_augmented_data["chat_history"].to_list(
                ),
                responses=responses,
                previous_documents=temporary_augmented_data["document"].to_list(
                ))

            # Step 4: Generate a new chat history dataset based on the questions asked: Could contained failed batch
            generated_chat_histories = []
            generated_chat_documents = []
            for chat_history, response, question, document in zip(
                temporary_augmented_data["chat_history"].to_list(),
                responses,
                questions,
                documents
            ):
                # Append the new conversation to the chat history
                new_chat_history = chat_history + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": question}
                ]
                
                # TODO: Remove the failed batch in the new chat history and documents

                generated_chat_histories.append(new_chat_history)
                generated_chat_documents.append(document)

            # Save the augmented data
            # TODO: Remove the failed batch in the temporary dataset
            augmented_data = augmented_data.vstack(
                copy.deepcopy(temporary_augmented_data))

            # Update temporary data for the next iteration
            temporary_augmented_data = pl.DataFrame({
                "chat_history": generated_chat_histories,
                "document": generated_chat_documents
            })

        return augmented_data
