"""
Pipeline function to generate augmented data for SLM fine-tuning based on given document knowledge
"""
import polars as pl
from tqdm import tqdm

from kolosal_backend.pipeline.parameter import KnowledgeParameter
from kolosal_backend.pipeline.pipeline_components import generate_conversation_starter, generate_conversations_response, comparison_score
from kolosal_backend.pipeline.knowledge import build_knowledge_instruction, build_chat_histories


def knowledge_pipeline(instruction: KnowledgeParameter) -> pl.DataFrame:
    augmented_data = pl.DataFrame(schema={
        "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
        "document": pl.Utf8,
        "slm_response": pl.Utf8,
        "llm_response": pl.Utf8,
        "scores": pl.Int64
    })

    temproary_augmented_data = pl.DataFrame(schema={
        "chat_history": pl.List(pl.Struct([pl.Field("role", pl.Utf8), pl.Field("content", pl.Utf8)])),
        "document": pl.Utf8
    })

    # Step 1: Generate conversation starter question based on the given documents
    for document in instruction.documents:
        built_knowledge_instructions = build_knowledge_instruction(instruction=instruction.conversation_starter_instruction,
                                                                   document=document)

        chat_histories_document = generate_conversation_starter(llm=instruction.llm_model,
                                                                num_instructions=instruction.conversation_starter_count,
                                                                instruction=built_knowledge_instructions,
                                                                system_prompt=None)

        document_data = [document] * len(chat_histories_document)

        temproary_augmented_data = temproary_augmented_data.vstack(pl.DataFrame({
            "chat_history": chat_histories_document,
            "document": document_data
        }))

    # Loop the conversation according to the instruction max conversation times to generate augmented data
    for i in tqdm(range(instruction.max_conversations)):
        # Step 2 Generate SLM and LLM response
        built_chat_histories = build_chat_histories(instruction=instruction.conversation_personalization_instruction,
                                                    documents=temproary_augmented_data["document"].to_list(
                                                    ),
                                                    chat_histories=temproary_augmented_data["chat_history"].to_list())

        slm_responses = generate_conversations_response(instruction.slm_model,
                                                        chat_histories=built_chat_histories)
        llm_responses = generate_conversations_response(instruction.llm_model,
                                                        chat_histories=built_chat_histories)

        temproary_augmented_data = temproary_augmented_data.with_columns(
            pl.Series("slm_response", slm_responses),
            pl.Series("llm_response", llm_responses)
        )

        # Step 3 Generate responses score
        scores = comparison_score(llm=instruction.llm_model,
                                  chat_histories=temproary_augmented_data["chat_history"].to_list(
                                  ),
                                  llm_responses=llm_responses,
                                  slm_responses=slm_responses)

        temproary_augmented_data = temproary_augmented_data.with_columns(
            pl.Series("scores", scores)
        )
        
        # Step 4 Generate a followup question based on the chat history and Document
        
    return temproary_augmented_data
