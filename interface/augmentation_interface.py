"""UI Kolosal Augmentation"""
import os
import io
import streamlit as st
import pandas as pd

from dotenv import load_dotenv

from distilabel.llms import AzureOpenAILLM, OpenAILLM, AnthropicLLM
from kolosal_plane import SimpleKnowledge  # Ensure the module name is correct

# Initialize session state variables if they don't exist
if "llm" not in st.session_state:
    st.session_state.llm = None
if "slm" not in st.session_state:
    st.session_state.slm = None
if "documents_df" not in st.session_state:
    st.session_state.documents_df = pd.DataFrame(columns=["Documents"])

load_dotenv()


def create_model(provider,
                 base_url=None,
                 api_key=None,
                 api_version=None,
                 model=None,
                 max_tokens=2048,
                 temperature=0.8):
    """
    Helper function to create LLM/SLM objects with validation.
    Note: For providers other than AzureOpenAI, `api_version` is optional.
    """
    try:
        if provider == "AzureOpenAI":
            # Validate required parameters for Azure
            if not all([base_url, api_key, api_version, model]):
                st.error(
                    "AzureOpenAI requires Base URL, API Key, API Version, and Deployment Name.")
                return None
            return AzureOpenAILLM(
                base_url=base_url,
                api_key=api_key,
                api_version=api_version,
                model=model,
                generation_kwargs={"max_new_tokens": max_tokens,
                                   "temperature": temperature}
            )
        elif provider == "OpenAI":
            if not api_key or not model:
                st.error("OpenAI requires API Key and Model Name.")
                return None
            return OpenAILLM(
                api_key=api_key,
                model=model,
                generation_kwargs={"max_new_tokens": max_tokens,
                                   "temperature": temperature}
            )
        elif provider == "Anthropics":
            if not api_key or not model:
                st.error("Anthropic requires API Key and Model Name.")
                return None
            return AnthropicLLM(
                api_key=api_key,
                model=model,
                generation_kwargs={"max_new_tokens": max_tokens,
                                   "temperature": temperature}
            )
        elif provider == "Fireworks":
            if not api_key or not model:
                st.error("Fireworks requires API Key and Model Name.")
                return None
            return OpenAILLM(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=api_key,
                model=model,
                generation_kwargs={"max_new_tokens": max_tokens,
                                   "temperature": temperature}
            )
        elif provider == "Custom":
            # Custom provider: only require Base URL, API Key and Model Name.
            if not base_url or not api_key or not model:
                st.error(
                    "Custom provider requires Base URL, API Key, and Model Name.")
                return None
            return OpenAILLM(
                base_url=base_url,
                api_key=api_key,
                model=model,
                generation_kwargs={"max_new_tokens": max_tokens,
                                   "temperature": temperature}
            )
        else:
            st.error(f"Unknown provider: {provider}")
            return None
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None


def setting_col_beta_test():
    st.subheader("Settings")
    st.text("Using OpenAI")

    # Setting LLM
    llm_config = {}
    llm_config['base_url'] = os.getenv("AZURE_OPENAI_ENDPOINT")
    llm_config['api_key'] = os.getenv("AZURE_OPENAI_API_KEY")
    llm_config['api_version'] = os.getenv("AZURE_API_VERSION")
    llm_config['model'] = "gpt-4o"
    
    
    
    max_tokens = st.number_input(
        "Max Tokens per Response", value=2048, min_value=1, key="max_tokens")
    llm_temperature = st.slider(
        "LLM Temperature", 0.0, 2.0, 0.8, key="llm_temp")
    
    st.divider()

    # --- Update Models ---
    if st.button("Update Settings"):
        # Ensure both required providers have been confirmed
        if "llm_option_confirmed" not in st.session_state or "slm_option_confirmed" not in st.session_state:
            st.error(
                "Please confirm both LLM and SLM Providers before updating settings.")
            return

        # Create the models using your `create_model` function
        st.session_state.llm = create_model(
            provider=st.session_state.llm_option_confirmed,
            max_tokens=max_tokens,
            temperature=llm_temperature,
            **llm_config
        )
        
        # As SLM is not used in beta test, let it be setting LLM
        st.session_state.slm = st.session_state.llm

        models_initialized = True
        if not st.session_state.llm or not st.session_state.slm:
            models_initialized = False
            st.error("Failed to initialize LLM and/or SLM – check parameters.")

        if models_initialized:
            st.success("Models initialized successfully!")

def setting_col():
    st.subheader("Settings")

    # --- LLM Provider Selection ---
    with st.form("llm_provider_form"):
        llm_option = st.radio(
            "Select LLM Provider:",
            ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks", "Custom"],
            key="llm_option"
        )
        llm_submit = st.form_submit_button("Confirm LLM Provider")
        if llm_submit:
            st.session_state.llm_option_confirmed = llm_option
            st.success(f"LLM Provider '{llm_option}' confirmed!")

    # Show LLM configuration inputs only after provider confirmation
    if "llm_option_confirmed" in st.session_state:
        st.markdown("#### LLM Configuration")
        llm_config = {}
        if st.session_state.llm_option_confirmed == "AzureOpenAI":
            llm_config['base_url'] = st.text_input(
                "LLM Base URL", key="llm_AzureOpenAI_base_url")
            llm_config['api_key'] = st.text_input(
                "LLM Azure API Key", type="password", key="llm_AzureOpenAI_api_key")
            llm_config['api_version'] = st.text_input(
                "LLM Azure API Version", key="llm_AzureOpenAI_api_version")
            llm_config['model'] = st.text_input(
                "LLM Deployment Name", key="llm_AzureOpenAI_model")
        elif st.session_state.llm_option_confirmed == "Custom":
            llm_config['base_url'] = st.text_input(
                "LLM Base URL", key="llm_Custom_base_url")
            llm_config['api_key'] = st.text_input(
                "LLM API Key", type="password", key="llm_Custom_api_key")
            llm_config['model'] = st.text_input(
                "LLM Model Name", key="llm_Custom_model")
        else:
            llm_config['api_key'] = st.text_input(
                f"LLM {st.session_state.llm_option_confirmed} API Key",
                type="password",
                key=f"llm_{st.session_state.llm_option_confirmed}_api_key"
            )
            llm_config['model'] = st.text_input(
                "LLM Model Name", key=f"llm_{st.session_state.llm_option_confirmed}_model")
    else:
        st.info("Please confirm an LLM Provider to see its configuration fields.")

    st.divider()

    # --- SLM Provider Selection ---
    with st.form("slm_provider_form"):
        slm_option = st.radio(
            "Select SLM Provider:",
            ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks", "Custom"],
            key="slm_option"
        )
        slm_submit = st.form_submit_button("Confirm SLM Provider")
        if slm_submit:
            st.session_state.slm_option_confirmed = slm_option
            st.success(f"SLM Provider '{slm_option}' confirmed!")

    # Show SLM configuration inputs only after provider confirmation
    if "slm_option_confirmed" in st.session_state:
        st.markdown("#### SLM Configuration")
        slm_config = {}
        if st.session_state.slm_option_confirmed == "AzureOpenAI":
            slm_config['base_url'] = st.text_input(
                "SLM Base URL", key="slm_AzureOpenAI_base_url")
            slm_config['api_key'] = st.text_input(
                "SLM Azure API Key", type="password", key="slm_AzureOpenAI_api_key")
            slm_config['api_version'] = st.text_input(
                "SLM Azure API Version", key="slm_AzureOpenAI_api_version")
            slm_config['model'] = st.text_input(
                "SLM Deployment Name", key="slm_AzureOpenAI_model")
        elif st.session_state.slm_option_confirmed == "Custom":
            slm_config['base_url'] = st.text_input(
                "SLM Base URL", key="slm_Custom_base_url")
            slm_config['api_key'] = st.text_input(
                "SLM API Key", type="password", key="slm_Custom_api_key")
            slm_config['model'] = st.text_input(
                "SLM Model Name", key="slm_Custom_model")
        else:
            slm_config['api_key'] = st.text_input(
                f"SLM {st.session_state.slm_option_confirmed} API Key",
                type="password",
                key=f"slm_{st.session_state.slm_option_confirmed}_api_key"
            )
            slm_config['model'] = st.text_input(
                "SLM Model Name", key=f"slm_{st.session_state.slm_option_confirmed}_model")
    else:
        st.info("Please confirm an SLM Provider to see its configuration fields.")

    st.divider()

    # --- Optional Thinking Model Configuration ---
    enable_tm = st.toggle("Enable Thinking Model (optional)", key="enable_tm")
    # Initialize an empty dictionary for Thinking Model config
    tm_config = {}
    if enable_tm:
        with st.form("tm_provider_form"):
            tm_option = st.radio(
                "Select Thinking Model Provider:",
                ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks", "Custom"],
                key="tm_option"
            )
            tm_submit = st.form_submit_button(
                "Confirm Thinking Model Provider")
            if tm_submit:
                st.session_state.tm_option_confirmed = tm_option
                st.success(f"Thinking Model Provider '{tm_option}' confirmed!")

        # Show Thinking Model configuration inputs only after provider confirmation
        if "tm_option_confirmed" in st.session_state:
            st.markdown("#### Thinking Model Configuration")
            if st.session_state.tm_option_confirmed == "AzureOpenAI":
                tm_config['base_url'] = st.text_input(
                    "Thinking Model Base URL", key="tm_AzureOpenAI_base_url")
                tm_config['api_key'] = st.text_input(
                    "Thinking Model Azure API Key", type="password", key="tm_AzureOpenAI_api_key")
                tm_config['api_version'] = st.text_input(
                    "Thinking Model Azure API Version", key="tm_AzureOpenAI_api_version")
                tm_config['model'] = st.text_input(
                    "Thinking Model Deployment Name", key="tm_AzureOpenAI_model")
            elif st.session_state.tm_option_confirmed == "Custom":
                tm_config['base_url'] = st.text_input(
                    "Thinking Model Base URL", key="tm_Custom_base_url")
                tm_config['api_key'] = st.text_input(
                    "Thinking Model API Key", type="password", key="tm_Custom_api_key")
                tm_config['model'] = st.text_input(
                    "Thinking Model Model Name", key="tm_Custom_model")
            else:
                tm_config['api_key'] = st.text_input(
                    f"Thinking Model {st.session_state.tm_option_confirmed} API Key",
                    type="password",
                    key=f"tm_{st.session_state.tm_option_confirmed}_api_key"
                )
                tm_config['model'] = st.text_input(
                    "Thinking Model Model Name", key=f"tm_{st.session_state.tm_option_confirmed}_model")
            tm_temperature = st.slider(
                "Thinking Model Temperature", 0.0, 2.0, 0.8, key="tm_temp"
            )
        else:
            st.info(
                "Please confirm a Thinking Model Provider to see its configuration fields.")

    st.divider()

    # --- Model Parameters (always visible) ---
    max_tokens = st.number_input(
        "Max Tokens per Response", value=2048, min_value=1, key="max_tokens")
    llm_temperature = st.slider(
        "LLM Temperature", 0.0, 2.0, 0.8, key="llm_temp")
    slm_temperature = st.slider(
        "SLM Temperature", 0.0, 2.0, 0.8, key="slm_temp")

    st.divider()

    # --- Update Models ---
    if st.button("Update Settings"):
        # Ensure both required providers have been confirmed
        if "llm_option_confirmed" not in st.session_state or "slm_option_confirmed" not in st.session_state:
            st.error(
                "Please confirm both LLM and SLM Providers before updating settings.")
            return

        # Create the models using your `create_model` function
        st.session_state.llm = create_model(
            provider=st.session_state.llm_option_confirmed,
            max_tokens=max_tokens,
            temperature=llm_temperature,
            **llm_config
        )
        st.session_state.slm = create_model(
            provider=st.session_state.slm_option_confirmed,
            max_tokens=max_tokens,
            temperature=slm_temperature,
            **slm_config
        )

        models_initialized = True
        if not st.session_state.llm or not st.session_state.slm:
            models_initialized = False
            st.error("Failed to initialize LLM and/or SLM – check parameters.")

        # Initialize the optional Thinking Model only if enabled
        if enable_tm:
            if "tm_option_confirmed" not in st.session_state:
                st.error(
                    "Please confirm Thinking Model Provider before updating settings.")
                return
            # tm_temperature was set only if a provider was confirmed
            st.session_state.thinking = create_model(
                provider=st.session_state.tm_option_confirmed,
                max_tokens=max_tokens,
                temperature=tm_temperature,
                **tm_config
            )
            if not st.session_state.thinking:
                models_initialized = False
                st.error("Failed to initialize Thinking Model – check parameters.")
            else:
                st.success("Thinking Model initialized successfully!")

        if models_initialized:
            st.success("Models initialized successfully!")


def prompt_col():
    st.subheader("Prompts")
    with st.form("prompts_form"):
        conversation_starter_instruction = st.text_area(
            "Conversation Starter Topic",
            height=200,
            key="starter_inst"
        )
        conversation_personalization_instruction = st.text_area(
            "Personalization Instructions",
            height=200,
            key="personal_inst"
        )
        system_prompt = st.text_area(
            "System Prompt",
            height=200,
            key="sys_prompt"
        )
        if st.form_submit_button("Update Prompts"):
            st.session_state.conversation_starter_instruction = conversation_starter_instruction
            st.session_state.conversation_personalization_instruction = conversation_personalization_instruction
            st.session_state.system_prompt = system_prompt


def documents_col():
    st.subheader("Documents")
    uploaded_file = st.file_uploader(
        "Upload CSV (optional)",
        type=["csv"],
        help="Upload a CSV file. We'll use the first column and rename it to 'Documents'",
        key="csv_upload"
    )

    if st.button("Load CSV", key="load_csv_button"):
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if not uploaded_df.empty:
                    first_column = uploaded_df.iloc[:, 0]
                    st.session_state.documents_df = pd.DataFrame(
                        {"Documents": first_column})
                    st.success(
                        f"Loaded {len(st.session_state.documents_df)} documents from CSV")
                else:
                    st.error("Uploaded CSV is empty")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        else:
            st.error("Please upload a CSV file first")

    st.data_editor(
        st.session_state.documents_df,
        num_rows="dynamic",
        column_config={"Documents": "Document"},
        height=400,
        key="documents_editor",
        use_container_width=True
    )


def augmentation_interface():
    st.write("Synthetic data generation using LLM for fine-tuning SLM")

    # Create three side-by-side columns: Settings, Prompts, Documents
    col_settings, col_prompts, col_documents = st.columns(3)

    # --------------------
    # Left Column: Settings (LLM/SLM configuration)
    # --------------------
    with col_settings:
        # Use the beta test
        setting_col_beta_test()

    # --------------------
    # Middle Column: Prompts
    # --------------------
    with col_prompts:
        prompt_col()

    # --------------------
    # Right Column: Documents (CSV uploader and Data Editor)
    # --------------------
    with col_documents:
        documents_col()

    # --------------------
    # Below the Columns: Additional Parameters and Data Generation
    # --------------------
    st.divider()
    conv_count = st.number_input(
        "Conversations per Document", 1, 64, 8, key="conv_count")
    max_conv_length = st.number_input(
        "Max Messages per Conversation", 1, 32, 8, key="max_conv_length")
    batch_size = st.number_input("Batch Size", 1, 256, 16, key="batch_size")

    if st.button("Generate Synthetic Data", key="generate_data"):
        if not st.session_state.llm or not st.session_state.slm:
            st.error("Configure models in Settings first!")
            return

        # Use the DataFrame stored in session_state as the source of documents
        docs = []
        if "Documents" in st.session_state.documents_df.columns:
            docs = st.session_state.documents_df["Documents"].dropna().tolist()
        if not docs:
            st.error("Add at least one document!")
            return

        try:
            pipeline = SimpleKnowledge(
                conversation_starter_instruction=st.session_state.get(
                    "conversation_starter_instruction", ""),
                conversation_personalization_instruction=st.session_state.get(
                    "conversation_personalization_instruction", ""),
                system_prompt=st.session_state.get("sys_prompt", ""),
                llm_model=st.session_state.llm,
                slm_model=st.session_state.slm,
                thinking_model=st.session_state.thinking,
                conversation_starter_count=conv_count,
                max_conversations=max_conv_length,
                batch_size=batch_size,
                documents=docs
            )

            with st.spinner("Generating data..."):
                response = pipeline.augmentate()

            # Convert the response to a dictionary and then to JSON
            json_buffer = io.StringIO()
            response.write_json(json_buffer)  # Use the same working method
            json_content = json_buffer.getvalue()

            st.download_button(
                "Download Dataset",
                data=json_content,
                file_name="synthetic_data.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
