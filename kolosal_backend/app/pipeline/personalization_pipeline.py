import polars as pl
from tqdm import tqdm

from kolosal_backend.app.pipeline.parameter import PersonalizationParameter
from kolosal_backend.app.pipeline.personalization import generate_conversation_starter, generate_conversations_response, comparison_score, generate_next_conversation


def personalization_pipeline(instruction: PersonalizationParameter) -> dict:
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
    temproary_augmented_data = pl.DataFrame({
        "chat_history": generate_conversation_starter(llm=instruction.llm_model, num_instructions=instruction.conversation_starter_count, instruction=instruction.conversation_starter_instruction, system_prompt=instruction.conversation_personalization_instruction)
    })

    # Loop the conversation n times to generate augmented data
    for i in tqdm(range(instruction.max_conversations)):
        # Step 2 Generate SLM and LLM response
        slm_responses = generate_conversations_response(llm=instruction.slm_model, chat_histories=temproary_augmented_data["chat_history"].to_list())
        llm_responses = generate_conversations_response(llm=instruction.llm_model, chat_histories=temproary_augmented_data["chat_history"].to_list())

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

        # Step 4 Generate a followup question based on the chat history
        slm_questions = generate_next_conversation(llm=instruction.llm_model, chat_histories=temproary_augmented_data["chat_history"].to_list(), responses=slm_responses)
        llm_questions = generate_next_conversation(llm=instruction.llm_model, chat_histories=temproary_augmented_data["chat_history"].to_list(), responses=llm_responses)

        # Step 5 Geneare a new chat history dataset based on the questions asked
        generated_chat_histories = []
        for chat_history, slm_response, llm_response, slm_question, llm_question in zip(temproary_augmented_data["chat_history"].to_list(), slm_responses, llm_responses, slm_questions, llm_questions):
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
        augmented_data = augmented_data.vstack(temproary_augmented_data)
        temproary_augmented_data = pl.DataFrame({
            "chat_history": generated_chat_histories
        })

    return augmented_data
