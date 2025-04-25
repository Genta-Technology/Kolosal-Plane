"""Prompt Interface for Data Augmentation"""
import streamlit as st


def knowledge_prompt_interface():
    """
    Renders a Streamlit interface for configuring knowledge augmentation prompts.
    This function displays a form with text areas for three different prompt components:
    - Conversation Starter Topic: Instructions for initiating conversations
    - Personalization Instructions: Guidelines for personalizing responses
    - System Prompt: Base instructions for the AI system
    The function loads existing values from st.session_state.knowledge_augmentation_config
    and updates the same configuration when the form is submitted.
    Returns:
        None: This function modifies st.session_state directly and renders Streamlit UI elements.
    """

    # Load the current configuration from session state
    conversation_starter_instruction = st.session_state.knowledge_augmentation_config.get(
        "conversation_starter_instruction", "")

    conversation_personalization_instruction = st.session_state.knowledge_augmentation_config.get(
        "conversation_personalization_instruction", "")

    system_prompt = st.session_state.knowledge_augmentation_config.get(
        "system_prompt", "")

    st.subheader("Prompts")

    # Example prompts
    if st.button("Load example prompts", key="load_example_prompts_knowledge"):
        st.session_state.knowledge_augmentation_config[
            "conversation_starter_instruction"] = "Act like an Ivy Leauge students that is passionate in astronomy, generate questions that would be asked to a professor based on the given topic."
        st.session_state.knowledge_augmentation_config[
            "conversation_personalization_instruction"] = "Answer the questions in a friendly manner, as if you are talking to a friend."
        st.session_state.knowledge_augmentation_config[
            "system_prompt"] = "You are an Ivy League professor, answer the questions in a friendly manner, as if you are teaching passionatly about astronomy"

    with st.form("prompts_form"):
        conversation_starter_instruction = st.text_area(
            "Conversation Starter Topic",
            value=conversation_starter_instruction,
            height=200,
            key="starter_inst"
        )
        conversation_personalization_instruction = st.text_area(
            "Personalization Instructions",
            value=conversation_personalization_instruction,
            height=200,
            key="personal_inst"
        )
        system_prompt = st.text_area(
            "System Prompt",
            value=system_prompt,
            height=200,
            key="sys_prompt"
        )
        if st.form_submit_button("Update Prompts"):
            st.session_state.knowledge_augmentation_config[
                "conversation_starter_instruction"] = conversation_starter_instruction
            st.session_state.knowledge_augmentation_config[
                "conversation_personalization_instruction"] = conversation_personalization_instruction
            st.session_state.knowledge_augmentation_config["system_prompt"] = system_prompt


def embedding_prompt_interface():
    """
    Streamlit interface for configuring embedding prompts in the data augmentation process.
    This function creates a form where users can:
    - View and edit the current instruction used for embedding-based data generation
    - Load example prompts with a single click
    - Update the instruction in the session state
    The function accesses and modifies the 'embeddings_augmentation_config' dictionary 
    in the Streamlit session state to maintain configuration persistence across reruns.
    Returns:
        None
    """

    # Load the current configuration from session state
    embedding_instruction = st.session_state.embeddings_augmentation_config.get(
        "instruction", "")

    # Example prompts
    if st.button("Load example prompts", key="load_example_prompts_embedding"):
        st.session_state.embeddings_augmentation_config[
            "instruction"] = "Act like an Ivy Leauge students that is passionate in astronomy, generate questions that would be asked to a professor based on the given topic."

    with st.form("instruction_form"):
        embedding_instruction = st.text_area(
            "Data Generation Instruction",
            value=embedding_instruction,
            height=200,
            key="instruction_form"
        )

        if st.form_submit_button("Update Prompts"):
            st.session_state.embeddings_augmentation_config[
                "instruction"] = embedding_instruction
