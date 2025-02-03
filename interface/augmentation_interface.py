import io
import streamlit as st
import pandas as pd

from distilabel.llms import AzureOpenAILLM, OpenAILLM, AnthropicLLM
from kolosal_plane import SimpleKnowledge

# Initialize session state for llm and slm if not already set
if "llm" not in st.session_state:
    st.session_state.llm = None
if "slm" not in st.session_state:
    st.session_state.slm = None
    
llm_option, llm_base_url, llm_api_key, llm_api_version, llm_model_name = None, None, None, None, None
slm_option, slm_base_url, slm_api_key, slm_api_version, slm_model_name = None, None, None, None, None

def create_model(provider, base_url, api_key, api_version, model, max_tokens, temperature):
    """Helper function to create LLM/SLM objects based on the provider."""
    if provider == "AzureOpenAI":
        return AzureOpenAILLM(
            base_url=base_url,
            api_key=api_key,
            api_version=api_version,
            model=model,
            generation_kwargs={"max_new_tokens": max_tokens, "temperature": temperature}
        )
    elif provider == "OpenAI":
        return OpenAILLM(
            api_key=api_key,
            model=model,
            generation_kwargs={"max_new_tokens": max_tokens, "temperature": temperature}
        )
    elif provider == "Anthropics":
        return AnthropicLLM(
            api_key=api_key,
            model=model,
            generation_kwargs={"max_new_tokens": max_tokens, "temperature": temperature}
        )
    elif provider == "Fireworks":
        return OpenAILLM(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
            model=model,
            generation_kwargs={"max_new_tokens": max_tokens, "temperature": temperature}
        )
    else:
        return None

def augmentation_interface():
    st.write("Synthetic data generation using LLM for fine-tuning SLM")
    col_settings, col_prompts, col_documents = st.columns(3)

    # ================================
    # Settings Column
    # ================================
    with col_settings:
        # Selecting LLM
        llm_options = ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"]
        llm_option = st.radio("Select LLM:", llm_options, key="llm_option")

        if llm_option == "AzureOpenAI":
            llm_base_url = st.text_input("LLM Base URL", key="llm_base_url")
            llm_api_key = st.text_input("LLM Azure API Key", key="llm_api_key")
            llm_api_version = st.text_input("LLM Azure API Version", key="llm_api_version")
            llm_model_name = st.text_input("LLM Model Name", key="llm_model_name")
        elif llm_option in ["OpenAI", "Anthropics", "Fireworks"]:
            llm_api_key = st.text_input(f"LLM {llm_option} API Key", key="llm_api_key")
            llm_model_name = st.text_input("LLM Model Name", key="llm_model_name")
            # For these providers, base_url and api_version are not used
            llm_base_url = ""
            llm_api_version = ""

        st.divider()

        # Selecting SLM
        slm_options = ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"]
        slm_option = st.radio("Select SLM:", slm_options, key="slm_option")

        if slm_option == "AzureOpenAI":
            slm_base_url = st.text_input("SLM Base URL", key="slm_base_url")
            slm_api_key = st.text_input("SLM Azure API Key", key="slm_api_key")
            slm_api_version = st.text_input("SLM Azure API Version", key="slm_api_version")
            slm_model_name = st.text_input("SLM Model Name", key="slm_model_name")
        elif slm_option in ["OpenAI", "Anthropics", "Fireworks"]:
            slm_api_key = st.text_input(f"SLM {slm_option} API Key", key="slm_api_key")
            slm_model_name = st.text_input("SLM Model Name", key="slm_model_name")
            slm_base_url = ""
            slm_api_version = ""

        st.divider()

        # Additional parameters
        max_tokens = st.number_input("Maximum Token Generated Per Chat", value=2048, step=128, key="max_tokens")
        llm_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.8, key="llm_temperature")
        slm_temperature = st.slider("SLM Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.8, key="slm_temperature")

        st.divider()

        # Update Settings Button: create and store the LLM/SLM objects in session_state
        if st.button("Update Setting"):
            st.session_state.llm = create_model(
                provider=llm_option,
                base_url=llm_base_url,
                api_key=llm_api_key,
                api_version=llm_api_version,
                model=llm_model_name,
                max_tokens=max_tokens,
                temperature=llm_temperature
            )
            st.session_state.slm = create_model(
                provider=slm_option,
                base_url=slm_base_url,
                api_key=slm_api_key,
                api_version=slm_api_version,
                model=slm_model_name,
                max_tokens=max_tokens,
                temperature=slm_temperature
            )

            if st.session_state.llm and st.session_state.slm:
                st.success(f"Successfully loaded LLM: {llm_option} and SLM: {slm_option}")
            else:
                st.error("Failed to load models. Check your input parameters.")

    # ================================
    # Prompts Column
    # ================================
    with col_prompts:
        conversation_starter_instruction = st.text_area(
            "Conversation Starter Topic (Describe what documents and goals of the conversation)",
            height=400, key="conversation_starter_instruction")
        conversation_personalization_instruction = st.text_area(
            "Conversation Personalization Instruction (Describe how would you like the AI to respond)",
            height=400, key="conversation_personalization_instruction")
        system_prompt = st.text_area("System Prompt of the AI", height=400, key="system_prompt")

    # ================================
    # Documents Column
    # ================================
    with col_documents:
        st.write("Insert your text document here")
        documents_df = pd.DataFrame(columns=["Documents"])
        edited_documents_df = st.data_editor(
            documents_df, num_rows="dynamic", use_container_width=True, height=400, key="documents_editor")

        st.divider()
        conversation_starter_count = st.number_input(
            "Total number of conversation per document", value=10, key="conversation_starter_count")
        max_conversations = st.number_input(
            "Total length of each conversation", value=10, key="max_conversations")
        batch_size = st.number_input(
            "Batch Size for generation requests", value=16, key="batch_size")

    st.divider()
    st.write("Warning: Ensure that you have properly added all of the parameters")

    # ================================
    # Augmentation Process
    # ================================
    if st.button("Augmentate Data"):
        if st.session_state.llm and st.session_state.slm:
            st.info("Start augmentation, please wait until the download button is available")
            knowledge_pipeline = SimpleKnowledge(
                conversation_starter_instruction=st.session_state.get("conversation_starter_instruction", ""),
                conversation_personalization_instruction=st.session_state.get("conversation_starter_instruction", ""),
                system_prompt=st.session_state.get("system_prompt", ""),
                llm_model=st.session_state.llm,
                slm_model=st.session_state.slm,
                conversation_starter_count=conversation_starter_count,
                max_conversations=max_conversations,
                batch_size=batch_size,
                documents=edited_documents_df["Documents"].tolist()
            )

            # Start data augmentation
            response = knowledge_pipeline.augmentate()

            # Convert the response to a dictionary and then to JSON
            json_buffer = io.StringIO()
            response.write_json(json_buffer)  # Use the same working method
            json_content = json_buffer.getvalue()

            # Provide a download button for the augmented data
            st.download_button(
                label="Download Augmented Dataset",
                data=json_content,
                file_name="augmented_dataset.json",
                mime="application/json",
            )
        else:
            st.error("LLM and SLM are not configured. Please update settings first.")