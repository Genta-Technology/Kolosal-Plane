import io
import streamlit as st
import pandas as pd

from distilabel.llms import AzureOpenAILLM, OpenAILLM, AnthropicLLM
from kolosal_plane import SimpleKnowledge  # Ensure correct module name

# Initialize session state for llm and slm if not already set
if "llm" not in st.session_state:
    st.session_state.llm = None
if "slm" not in st.session_state:
    st.session_state.slm = None

def create_model(provider, base_url, api_key, api_version, model, max_tokens, temperature):
    """Helper function to create LLM/SLM objects with validation."""
    try:
        if provider == "AzureOpenAI":
            # Validate required parameters for Azure
            if not all([base_url, api_key, api_version, model]):
                st.error("AzureOpenAI requires Base URL, API Key, API Version, and Deployment Name.")
                return None
            return AzureOpenAILLM(
                base_url=base_url,
                api_key=api_key,
                api_version=api_version,
                model=model,
                generation_kwargs={"max_tokens": max_tokens, "temperature": temperature}  # Fixed parameter
            )
        elif provider == "OpenAI":
            if not api_key or not model:
                st.error("OpenAI requires API Key and Model Name.")
                return None
            return OpenAILLM(
                api_key=api_key,
                model=model,
                generation_kwargs={"max_tokens": max_tokens, "temperature": temperature}  # Fixed parameter
            )
        elif provider == "Anthropics":
            if not api_key or not model:
                st.error("Anthropic requires API Key and Model Name.")
                return None
            return AnthropicLLM(
                api_key=api_key,
                model=model,
                generation_kwargs={"max_tokens": max_tokens, "temperature": temperature}  # Fixed parameter
            )
        elif provider == "Fireworks":
            if not api_key or not model:
                st.error("Fireworks requires API Key and Model Name.")
                return None
            return OpenAILLM(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=api_key,
                model=model,
                generation_kwargs={"max_tokens": max_tokens, "temperature": temperature}  # Fixed parameter
            )
        else:
            return None
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

def augmentation_interface():
    st.write("Synthetic data generation using LLM for fine-tuning SLM")
    
    # Using forms to prevent premature reruns
    with st.form("settings_form"):
        col_settings, col_prompts, col_documents = st.columns(3)

        # ================================
        # Settings Column
        # ================================
        with col_settings:
            # LLM Configuration
            llm_option = st.radio("Select LLM:", ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"], key="llm_option")
            llm_config = {}
            if llm_option == "AzureOpenAI":
                llm_config['base_url'] = st.text_input("LLM Base URL", key="llm_base_url")
                llm_config['api_key'] = st.text_input("LLM Azure API Key", type="password", key="llm_api_key")
                llm_config['api_version'] = st.text_input("LLM Azure API Version", key="llm_api_version")
                llm_config['model'] = st.text_input("LLM Deployment Name", key="llm_model_name")  # Corrected label
            else:
                llm_config['api_key'] = st.text_input(f"LLM {llm_option} API Key", type="password", key="llm_api_key")
                llm_config['model'] = st.text_input("LLM Model Name", key="llm_model_name")

            st.divider()

            # SLM Configuration
            slm_option = st.radio("Select SLM:", ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks"], key="slm_option")
            slm_config = {}
            if slm_option == "AzureOpenAI":
                slm_config['base_url'] = st.text_input("SLM Base URL", key="slm_base_url")
                slm_config['api_key'] = st.text_input("SLM Azure API Key", type="password", key="slm_api_key")
                slm_config['api_version'] = st.text_input("SLM Azure API Version", key="slm_api_version")
                slm_config['model'] = st.text_input("SLM Deployment Name", key="slm_model_name")  # Corrected label
            else:
                slm_config['api_key'] = st.text_input(f"SLM {slm_option} API Key", type="password", key="slm_api_key")
                slm_config['model'] = st.text_input("SLM Model Name", key="slm_model_name")

            st.divider()

            # Model Parameters
            max_tokens = st.number_input("Max Tokens per Response", value=2048, min_value=1, key="max_tokens")
            llm_temperature = st.slider("LLM Temperature", 0.0, 2.0, 0.8, key="llm_temp")
            slm_temperature = st.slider("SLM Temperature", 0.0, 2.0, 0.8, key="slm_temp")

            # Submit button for settings
            if st.form_submit_button("Update Settings"):
                st.session_state.llm = create_model(
                    provider=llm_option,
                    max_tokens=max_tokens,
                    temperature=llm_temperature,
                    **llm_config
                )
                st.session_state.slm = create_model(
                    provider=slm_option,
                    max_tokens=max_tokens,
                    temperature=slm_temperature,
                    **slm_config
                )
                if st.session_state.llm and st.session_state.slm:
                    st.success("Models initialized successfully!")
                else:
                    st.error("Failed to initialize models - check parameters")

        # ================================
        # Prompts Column
        # ================================
        with col_prompts:
            st.session_state.conversation_starter_instruction = st.text_area(
                "Conversation Starter Topic", 
                height=200,
                key="starter_inst"
            )
            st.session_state.conversation_personalization_instruction = st.text_area(
                "Personalization Instructions", 
                height=200,
                key="personal_inst"
            )
            st.session_state.system_prompt = st.text_area(
                "System Prompt", 
                height=200,
                key="sys_prompt"
            )

        # ================================
        # Documents Column
        # ================================
        with col_documents:
            st.write("Insert your text document here")
            
            # CSV file uploader
            uploaded_file = st.file_uploader(
                "Upload CSV (optional)", 
                type=["csv"], 
                help="Upload a CSV file. We'll use the first column and rename it to 'Documents'",
                key="csv_upload"
            )
            
            # Initialize documents dataframe
            documents_df = pd.DataFrame(columns=["Documents"])
            
            # Process uploaded CSV
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    if not uploaded_df.empty:
                        # Extract first column regardless of name and convert to Documents
                        first_column = uploaded_df.iloc[:, 0]
                        documents_df = pd.DataFrame({"Documents": first_column})
                        st.success(f"Loaded {len(documents_df)} documents from CSV")
                    else:
                        st.error("Uploaded CSV is empty")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
            
            # Data editor with CSV data or empty
            edited_documents_df = st.data_editor(
                documents_df,
                num_rows="dynamic",
                column_config={"Documents": "Document"},
                height=400,
                key="documents_editor",
                use_container_width=True
            )
            
            st.divider()
            conv_count = st.number_input("Conversations per Document", 1, 64, 8)
            max_conv_length = st.number_input("Max Messages per Conversation", 1, 32, 8)
            batch_size = st.number_input("Batch Size", 1, 256, 16)

    # ================================
    # Augmentation Execution
    # ================================
    if st.button("Generate Synthetic Data"):
        # Validate models
        if not st.session_state.llm or not st.session_state.slm:
            st.error("Configure models in Settings first!")
            return
        
        # Validate documents
        docs = []
        if not edited_documents_df.empty and "Documents" in edited_documents_df.columns:
            docs = edited_documents_df["Documents"].dropna().tolist()
        if not docs:
            st.error("Add at least one document!")
            return

        # Execute pipeline
        try:
            pipeline = SimpleKnowledge(
                conversation_starter_instruction=st.session_state.conversation_starter_instruction,
                conversation_personalization_instruction=st.session_state.conversation_personalization_instruction,  # Fixed key
                system_prompt=st.session_state.system_prompt,
                llm_model=st.session_state.llm,
                slm_model=st.session_state.slm,
                conversation_starter_count=conv_count,
                max_conversations=max_conv_length,
                batch_size=batch_size,
                documents=docs
            )
            
            with st.spinner("Generating data..."):
                df = pipeline.augmentate()  # Verify method name
                
            # Export results
            json_buf = io.StringIO()
            df.to_json(json_buf, orient="records")
            st.download_button(
                "Download Dataset",
                data=json_buf.getvalue(),
                file_name="synthetic_data.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")