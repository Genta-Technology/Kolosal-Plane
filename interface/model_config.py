"""Model Configuration Helper"""
import streamlit as st


def create_model_config(provider: str,
                        model: str,
                        api_key: str,
                        base_url: str = None,
                        api_version: str = None,
                        temperature: float = 0.8,
                        max_tokens: float = 2048):
    """
    Helper function to create model configuration for API requests.
    Maps UI provider names to API expected values.
    """
    # Map UI provider names to API expected values
    provider_mapping = {
        "AzureOpenAI": "azure_openai",
        "OpenAI": "openai",
        "Anthropics": "anthropic",
        "Fireworks": "fireworks",
        "Custom": "openai"  # Custom uses OpenAI-compatible API
    }

    api_provider = provider_mapping.get(provider)
    if not api_provider:
        st.error(f"Unknown provider: {provider}")
        return None

    # Validate required parameters
    if not model or not api_key:
        st.error(f"Model name and API key are required for {provider}")
        return None

    # Additional validation for AzureOpenAI
    if provider == "AzureOpenAI" and not all([base_url, api_version]):
        st.error("AzureOpenAI requires Base URL and API Version")
        return None

    # Additional validation for Custom provider
    if provider == "Custom" and not base_url:
        st.error("Custom provider requires Base URL")
        return None

    model_config = {
        "model_provider": api_provider,
        "model_name": model,
        "api_key": api_key,
        "model_parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }

    # Add optional parameters if provided
    if base_url:
        model_config["base_url"] = base_url
    if api_version:
        model_config["api_version"] = api_version

    return model_config


def model_config_interface():
    """
    Creates an interactive Streamlit interface for configuring Language Learning Models (LLM) 
    and Thinking Language Models (TLM).
    Interface Features:
    - Split into two columns for LLM and TLM configuration
    - Allows selection of model providers (AzureOpenAI, OpenAI, Anthropics, Fireworks, Custom)
    - Provider-specific configuration fields appear after confirmation
    - TLM can be toggled on/off
    Provider Configuration:
    - AzureOpenAI: Base URL, API Key, API Version, Deployment Name
    - Custom: Base URL, API Key, Model Name
    - Others (OpenAI, Anthropics, Fireworks): API Key and Model Name
    Stores selected configurations in Streamlit session state for use in other parts of the application.
    Returns:
        None
    """

    # Create two columns for LLM and TLM configuration
    col1, col2 = st.columns(2)

    # --- LLM Provider Selection (Left Column) ---
    with col1:
        st.markdown("### LLM Configuration")
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
        llm_config = {}
        if "llm_option_confirmed" in st.session_state:
            provider = st.session_state.llm_option_confirmed

            # Temperature and max tokens inputs
            col_temp, col_tokens = st.columns(2)
            with col_temp:
                temperature = st.slider(
                    "Temperature", 0.0, 1.0, 0.8, 0.1, key="llm_temperature")
            with col_tokens:
                max_tokens = st.number_input(
                    "Max Tokens", 1, 4096, 2048, key="llm_max_tokens")

            # Provider-specific fields
            if provider == "AzureOpenAI":
                base_url = st.text_input(
                    "LLM Base URL", key="llm_AzureOpenAI_base_url")
                api_key = st.text_input(
                    "LLM Azure API Key", type="password", key="llm_AzureOpenAI_api_key")
                api_version = st.text_input(
                    "LLM Azure API Version", key="llm_AzureOpenAI_api_version")
                model = st.text_input(
                    "LLM Deployment Name", key="llm_AzureOpenAI_model")

                # Create model config using the helper function
                llm_config = create_model_config(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    api_version=api_version,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            elif provider == "Custom":
                base_url = st.text_input(
                    "LLM Base URL", key="llm_Custom_base_url")
                api_key = st.text_input(
                    "LLM API Key", type="password", key="llm_Custom_api_key")
                model = st.text_input("LLM Model Name", key="llm_Custom_model")

                # Create model config using the helper function
                llm_config = create_model_config(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            else:
                api_key = st.text_input(
                    f"LLM {provider} API Key",
                    type="password",
                    key=f"llm_{provider}_api_key"
                )
                model = st.text_input(
                    "LLM Model Name", key=f"llm_{provider}_model")

                # Create model config using the helper function
                llm_config = create_model_config(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        else:
            st.info("Please confirm an LLM Provider to see its configuration fields.")

    # --- TLM Provider Selection (Right Column) ---
    with col2:
        st.markdown("### TLM Configuration")

        # Toggle for enabling/disabling TLM
        enable_tlm = st.toggle(
            "Enable Thinking Language Model (TLM)", key="enable_tlm")

        tlm_config = {}
        if enable_tlm:
            with st.form("tlm_provider_form"):
                tlm_option = st.radio(
                    "Select TLM Provider:",
                    ["AzureOpenAI", "OpenAI", "Anthropics", "Fireworks", "Custom"],
                    key="tlm_option"
                )
                tlm_submit = st.form_submit_button("Confirm TLM Provider")
                if tlm_submit:
                    st.session_state.tlm_option_confirmed = tlm_option
                    st.success(f"TLM Provider '{tlm_option}' confirmed!")

            # Show TLM configuration inputs only after provider confirmation
            if "tlm_option_confirmed" in st.session_state:
                provider = st.session_state.tlm_option_confirmed

                # Temperature and max tokens inputs
                col_temp, col_tokens = st.columns(2)
                with col_temp:
                    temperature = st.slider(
                        "Temperature", 0.0, 1.0, 0.8, 0.1, key="tlm_temperature")
                with col_tokens:
                    max_tokens = st.number_input(
                        "Max Tokens", 1, 4096, 2048, key="tlm_max_tokens")

                # Provider-specific fields
                if provider == "AzureOpenAI":
                    base_url = st.text_input(
                        "TLM Base URL", key="tlm_AzureOpenAI_base_url")
                    api_key = st.text_input(
                        "TLM Azure API Key", type="password", key="tlm_AzureOpenAI_api_key")
                    api_version = st.text_input(
                        "TLM Azure API Version", key="tlm_AzureOpenAI_api_version")
                    model = st.text_input(
                        "TLM Deployment Name", key="tlm_AzureOpenAI_model")

                    # Create model config using the helper function
                    tlm_config = create_model_config(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        api_version=api_version,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                elif provider == "Custom":
                    base_url = st.text_input(
                        "TLM Base URL", key="tlm_Custom_base_url")
                    api_key = st.text_input(
                        "TLM API Key", type="password", key="tlm_Custom_api_key")
                    model = st.text_input(
                        "TLM Model Name", key="tlm_Custom_model")

                    # Create model config using the helper function
                    tlm_config = create_model_config(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                else:
                    api_key = st.text_input(
                        f"TLM {provider} API Key",
                        type="password",
                        key=f"tlm_{provider}_api_key"
                    )
                    model = st.text_input(
                        "TLM Model Name", key=f"tlm_{provider}_model")

                    # Create model config using the helper function
                    tlm_config = create_model_config(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            else:
                st.info(
                    "Please confirm a TLM Provider to see its configuration fields.")
        else:
            st.info("TLM is disabled. Toggle the switch above to enable.")

    save_configuration = st.button("Confirm LLM and TLM Settings")
    if save_configuration:
        # Update session state with the configurations
        st.session_state.knowledge_augmentation_config["llm_config"] = llm_config
        st.session_state.knowledge_augmentation_config["tlm_config"] = tlm_config

        st.session_state.embeddings_augmentation_config["llm_config"] = llm_config
        st.session_state.embeddings_augmentation_config["tlm_config"] = tlm_config

        st.success("Model Providers confirmed!")
