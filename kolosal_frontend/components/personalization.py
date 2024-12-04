"""
Personalization UI
"""

import streamlit as st

def personalization_ui():
    st.title("Personalized your AI with your own preferences")
    chat_interface_tab, manual_interface_tab = st.tabs(["Chat Based Interface", "Manual Interface"])
    
    with chat_interface_tab:
        chat_interface_tab_ui()
    
    with manual_interface_tab:
        manual_interface_tab_ui()


def chat_interface_tab_ui():
    """
    Renders the chat interface tab for AI Assisted Chat Preferences in a Streamlit application.
    This function creates three main containers: header, chat history, and input. It displays a header
    section with a title and caption, a chat history section that shows all previous messages, and an
    input section where users can type their messages and interact with the AI.
    The chat history section iterates over the messages stored in `st.session_state.personalization_message`
    and displays them accordingly. The input section includes a chat input box for users to type their
    messages and a button to clear the chat history.
    When a user submits a message, it is added to the session state, and a simulated bot response is
    generated (to be replaced with an actual API call). The chat interface is then rerun to update the
    display.
    The clear chat button resets the chat history in the session state and reruns the interface.
    Note:
        - The bot response is currently hardcoded as "hello" and should be replaced with an actual API call.
        - The `st.rerun()` function is used to refresh the chat display after each user input or chat clear action.
    """
    
    # Create main containers
    header_container = st.container()
    chat_container = st.container(height=600, border=False)
    input_container = st.container()

    # Header section
    with header_container:
        st.header("AI Assisted Chat Preferences")
        st.caption("Talk to AI on how you want to personalize your AI")
        
    # Chat history section
    with chat_container:
        # Display all messages
        for msg in st.session_state.personalization_message:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input section - always at bottom
    with input_container:
        if prompt := st.chat_input("Type your message here...", key="chat_input_personalization"):
            # Add user message to state
            st.session_state.personalization_message.append({
                "role": "user",
                "content": prompt
            })

            # Simulate bot response, TODO: Replace with actual API agent call
            response = "hello"  # Replace with actual API call

            # Add assistant response to state
            st.session_state.personalization_message.append({
                "role": "assistant",
                "content": response
            })

            # Rerun to update the chat display
            st.rerun()

        if st.button("Clear chat", key="clear_chat_personalization"):
            st.session_state.personalization_message = []
            st.rerun()

def manual_interface_tab_ui():
    """
    Renders the UI for manually editing model preferences in a Streamlit application.
    This function creates a user interface with several input fields and sliders
    that allow users to manually edit their model preferences. The preferences
    include the AI system prompt, user prompt, conversation starter instruction,
    maximum conversation length, and maximum conversation start.
    The following elements are included in the UI:
    - A text area for editing the AI system prompt.
    - A text area for editing the modified user prompt.
    - A text area for editing the conversation starter instruction.
    - A slider for setting the maximum conversation length.
    - A slider for setting the maximum conversation start.
    The values entered by the user are stored in the Streamlit session state.
    """
    
    st.write("Manually edit your model preferences here")
    
    st.session_state.chatbot_system = st.text_area("Your AI System Prompt", st.session_state.chatbot_system)
    
    st.session_state.user_prompt = st.text_area("Your Modified User Prompt", st.session_state.user_prompt)
    
    st.session_state.conversation_starter_instruction = st.text_area("Conversation Starter Instruction", st.session_state.conversation_starter_instruction)
    
    st.session_state.max_conversation_length = st.slider("Max Conversation Length", value=st.session_state.max_conversation_length)
    
    st.session_state.max_conversation_start = st.slider("Max Conversation Start", value=st.session_state.max_conversation_start)