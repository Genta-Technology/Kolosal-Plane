"""Chatbot preview interface using Streamlit."""
import streamlit as st


def chatbot_preview_ui():
    """
    Renders a chatbot user interface using Streamlit.
    This function creates a simple chatbot interface with three main sections:
    a header, a chat history display, and an input area for user messages.
    It maintains the chat history using Streamlit's session state and simulates
    bot responses.
    The interface includes:
    - A header section with a title and caption.
    - A chat history section that displays all messages exchanged between the user and the bot.
    - An input section where the user can type their message and send it to the bot.
    The function also includes a "Clear chat" button to reset the chat history.
    Note:
    - st.session_state.messages: A list of dictionaries containing the chat messages. Each dictionary has two keys:
        - "role": The role of the message sender ("user" or "assistant").
        - "content": The content of the message.
    Streamlit Components:
    - st.container(): Creates a container for grouping elements.
    - st.header(): Displays a header.
    - st.caption(): Displays a caption.
    - st.chat_message(): Displays a chat message.
    - st.markdown(): Renders Markdown text.
    - st.chat_input(): Creates an input box for typing messages.
    - st.button(): Creates a button.
    Example:
        To use this function, simply call it within a Streamlit app:
        chatbot_ui()
    """

    # Create main containers
    header_container = st.container()
    chat_container = st.container(height=800, border=False)
    input_container = st.container()

    # Header section
    with header_container:
        st.header("Chatbot Preview")
        st.caption("Preview of your own personalized chatbot")

    # Chat history section
    with chat_container:
        # Display all messages
        for msg in st.session_state.messages_preview:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input section - always at bottom
    with input_container:
        if prompt := st.chat_input("Type your message here...", key="chat_input"):
            # Add user message to state
            st.session_state.messages_preview.append({
                "role": "user",
                "content": prompt
            })

            # Simulate bot response, TODO: Replace with actual API call
            response = "hello"  # Replace with actual API call

            # Add assistant response to state
            st.session_state.messages_preview.append({
                "role": "assistant",
                "content": response
            })

            # Rerun to update the chat display
            st.rerun()

        if st.button("Clear chat", key="clear_chat"):
            st.session_state.messages_preview = []
            st.rerun()
