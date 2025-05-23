"""Setting Interface for Kolosal Plane"""

import streamlit as st


def knowledge_configuration_interface():
    """
    Renders the configuration interface for knowledge augmentation settings.
    This function displays and manages user inputs for configuring the knowledge augmentation process:
    - Conversation starter count: Number of initial conversation starters to generate
    - Max conversation length: Maximum number of turns in the generated conversations
    - Batch size: Number of examples to process at once during augmentation
    The function retrieves default values from the session state and updates the configuration
    when the form is submitted.
    Returns:
        None
    Side effects:
        - Updates st.session_state.knowledge_augmentation_config with the new values
        - Renders UI elements in the Streamlit application
    """

    conversation_starter_count = st.session_state.knowledge_augmentation_config.get(
        "conversation_starter_count", 4)
    max_conversation_length = st.session_state.knowledge_augmentation_config.get(
        "max_conversation_length", 4)
    batch_size = st.session_state.knowledge_augmentation_config.get(
        "batch_size", 16)

    st.subheader("Augmentation Configuration")
    with st.form("augmentation_config_form"):
        conversation_starter_count = st.number_input(
            "Conversation Starter Count",
            value=conversation_starter_count,
            min_value=1,
            max_value=100,
            step=1,
            key="starter_count"
        )
        max_conversation_length = st.number_input(
            "Max Conversation Length",
            value=max_conversation_length,
            min_value=1,
            max_value=100,
            step=1,
            key="max_conv_len"
        )
        batch_size = st.number_input(
            "Batch Size",
            value=batch_size,
            min_value=8,
            max_value=256,
            step=8,
            key="batch_size"
        )
        if st.form_submit_button("Update Configuration"):
            st.session_state.knowledge_augmentation_config[
                "conversation_starter_count"] = conversation_starter_count
            st.session_state.knowledge_augmentation_config[
                "max_conversation_length"] = max_conversation_length
            st.session_state.knowledge_augmentation_config["batch_size"] = batch_size


def embedding_configuration_interface():
    """
    Renders the configuration interface for embedding augmentation settings.
    This function displays and manages user inputs for configuring the embedding augmentation process:
    - Question per document: Number of questions to generate for each document
    - Batch size: Number of examples to process at once during augmentation
    The function retrieves default values from the session state and updates the configuration
    when the form is submitted.
    Returns:
        None
    Side effects:
        - Updates st.session_state.embeddings_augmentation_config with the new values
        - Renders UI elements in the Streamlit application
    """

    question_per_document = st.session_state.embeddings_augmentation_config.get(
        "question_per_document", 100)
    batch_size = st.session_state.embeddings_augmentation_config.get(
        "batch_size", 16)

    st.subheader("Augmentation Configuration")
    with st.form("embedding_config_form"):
        question_per_document = st.number_input(
            "Question Per Document",
            value=question_per_document,
            min_value=1,
            max_value=100,
            step=1,
            key="question_per_doc"
        )
        batch_size = st.number_input(
            "Batch Size",
            value=batch_size,
            min_value=8,
            max_value=256,
            step=8,
            key="batch_size"
        )
        if st.form_submit_button("Update Configuration"):
            st.session_state.embeddings_augmentation_config[
                "question_per_document"] = question_per_document
            st.session_state.embeddings_augmentation_config["batch_size"] = batch_size
