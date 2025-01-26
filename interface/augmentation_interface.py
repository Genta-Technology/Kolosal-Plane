import json

import streamlit as st
import pandas as pd

from distilabel.llms import AzureOpenAILLM, OpenAILLM, AnthropicLLM
from kolosal_plane import SimpleKnowledge


def augmentation_interface():
    st.write("Synthetic data generation using LLM for fine-tuning SLM")

    col_settings, col_prompts, col_documents = st.columns(3)
    llm, slm = None, None

    with col_settings:
        # Selecting LLM
        llm_options = ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"]
        llm_option = st.radio("Select LLM:", llm_options)

        if llm_option == "AzureOpenAI":
            # Azure OpenAI
            llm_base_url = st.text_input("LLM Base URL")
            llm_api_key = st.text_input("LLM Azure API Key")
            llm_api_version = st.text_input("LLM Azure API Version")
            llm_model = st.text_input("LLM Model Name")

        elif llm_option == "OpenAI":
            # OpenAI
            llm_api_key = st.text_input("LLM OpenAI API Key")
            llm_model = st.text_input("LLM Model Name")

        elif llm_option == "Anthropics":
            # Anthropics
            llm_api_key = st.text_input("LLM Anthropics API Key")
            llm_model = st.text_input("LLM Model Name")

        elif llm_option == "Fireworks":
            # Fireworks
            llm_api_key = st.text_input("LLM Fireworks API Key")
            llm_model = st.text_input("LLM Model Name")

        st.divider()
        # Selecting SLM
        slm_options = ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"]
        slm_option = st.radio("Select SLM:", slm_options)

        if slm_option == "AzureOpenAI":
            # Azure OpenAI
            slm_base_url = st.text_input("SLM Base URL")
            slm_api_key = st.text_input("SLM Azure API Key")
            slm_api_version = st.text_input("SLM Azure API Version")
            slm_model = st.text_input("SLM Model Name")

        elif slm_option == "OpenAI":
            # OpenAI
            slm_api_key = st.text_input("SLM OpenAI API Key")
            slm_model = st.text_input("SLM Model Name")

        elif slm_option == "Anthropics":
            # Anthropics
            slm_api_key = st.text_input("SLM Anthropics API Key")
            slm_model = st.text_input("SLM Model Name")

        elif slm_option == "Fireworks":
            # Fireworks
            slm_api_key = st.text_input("SLM Fireworks API Key")
            slm_model = st.text_input("SLM Model Name")

        st.divider()
        # Add Parameter
        max_tokens = st.number_input(
            "Maximum Token Generated Per Chat", value=2048, step=128)
        llm_temperature = st.slider(
            label="LLM Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.8)
        slm_temperature = st.slider(
            label="SLM Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.8)

        st.divider()
        # Finished Setting
        if st.button("Update Setting"):
            if llm_option and slm_option is not None:
                if llm_option == "AzureOpenAI":
                    llm = AzureOpenAILLM(
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                        api_version=llm_api_version,
                        model=llm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": llm_temperature
                        }
                    )

                elif llm_option == "OpenAI":
                    llm = OpenAILLM(
                        api_key=llm_api_key,
                        model=llm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": llm_temperature
                        }
                    )

                elif llm_option == "Anthropics":
                    llm = AnthropicLLM(
                        api_key=llm_api_key,
                        model=llm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": llm_temperature
                        }
                    )

                elif llm_option == "Fireworks":
                    llm = OpenAILLM(
                        base_url="https://api.fireworks.ai/inference/v1",
                        api_key=llm_api_key,
                        model=llm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": llm_temperature
                        }
                    )

                if slm_option == "AzureOpenAI":
                    slm = AzureOpenAILLM(
                        base_url=slm_base_url,
                        api_key=slm_api_key,
                        api_version=slm_api_version,
                        model=slm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": slm_temperature
                        }
                    )

                elif slm_option == "OpenAI":
                    slm = OpenAILLM(
                        api_key=slm_api_key,
                        model=slm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": slm_temperature
                        }
                    )

                elif slm_option == "Anthropics":
                    slm = AnthropicLLM(
                        api_key=slm_api_key,
                        model=slm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": slm_temperature
                        }
                    )

                elif slm_option == "Fireworks":
                    slm = OpenAILLM(
                        base_url="https://api.fireworks.ai/inference/v1",
                        api_key=slm_api_key,
                        model=slm_model,
                        generation_kwargs={
                            "max_new_tokens": max_tokens,
                            "temperature": slm_temperature
                        }
                    )

                st.write(
                    f"Sucessfully load LLM: {llm_option} and SLM: {slm_option}")

    with col_prompts:
        conversation_starter_instruction = st.text_area(
            "Conversation Starter Topic (Describe what documents and goals of the conversation)", height=400)
        conversation_personalization_instruction = st.text_area(
            "Conversation Personalization Instruction (Describe how would you like the AI to respond)", height=400)

    with col_documents:
        documents_df = pd.DataFrame(
            columns=["Documents"]
        )

        # Insert Document
        st.write("Insert your text document here")
        edited_documents_df = st.data_editor(
            documents_df, num_rows="dynamic", use_container_width=True, height=400)

        st.divider()
        # Document Augmentation Parameter
        conversation_starter_count = st.number_input(
            "Total number of conversation per document", value=10)
        max_conversations = st.number_input(
            "Total length of each conversation", value=10)

        # Augmentation Batch Size
        batch_size = st.number_input(
            "Batch Size for generation requests", value=16)

    st.divider()
    st.write("Warning: Ensure that you have properly added all of the parameters")
    if st.button("Augmentate Data") and (llm is not None) and (slm is not None):
        knowledge_pipeline = SimpleKnowledge(conversation_starter_instruction=conversation_starter_instruction,
                                             conversation_personalization_instruction=conversation_personalization_instruction,
                                             llm_model=llm,
                                             slm_model=slm,
                                             conversation_starter_count=conversation_starter_count,
                                             max_conversations=max_conversations,
                                             batch_size=batch_size,
                                             documents=edited_documents_df["Documents"].tolist())

        # Start Augmentation Data
        response = knowledge_pipeline.augmentate()

        # Assuming `response` contains the output from `knowledge_pipeline.augmentate()`
        # Convert response to a dictionary (or use appropriate method if not `.to_dict`)
        response_data = response.to_dict()

        # Convert the response to JSON format
        response_json = json.dumps(response_data, indent=4)

        # Add a download button in the Streamlit app
        st.download_button(
            label="Download Augmented Dataset",
            data=response_json,
            file_name="augmented_dataset.json",
            mime="application/json",
        )
