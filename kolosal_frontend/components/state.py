"""
Kolosal UI Session State Initial Load
"""

import streamlit as st


def initialized_state():
    """
    Initialize the session state with default values if they are not already set.
    This function checks for the presence of specific keys in the Streamlit session state
    and initializes them with default values if they do not exist. The keys and their
    default values are as follows:
    - "messages_preview": an empty list
    - "personalization_message": an empty list
    - "chatbot_system": a string indicating the chatbot's role
    - "user_prompt": a string template for user input
    - "conversation_starter_instruction": a string with instructions for starting a conversation
    This ensures that the session state has the necessary keys with appropriate default values
    for the application to function correctly.
    """

    if "messages_preview" not in st.session_state:
        st.session_state.messages_preview = []

    if "personalization_message" not in st.session_state:
        st.session_state.personalization_message = []

    if "chatbot_system" not in st.session_state:
        st.session_state.chatbot_system = "You are a helpful assistant that can answer question using LinkedIn terms"

    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = "{user_input}"

    if "conversation_starter_instruction" not in st.session_state:
        st.session_state.conversation_starter_instruction = "Start with the topic of university life"

    if "max_conversation_length" not in st.session_state:
        st.session_state.max_conversation_length = 1

    if "max_conversation_start" not in st.session_state:
        st.session_state.max_conversation_start = 5
