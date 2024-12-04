"""Main UI frontend for Kolosal AI"""

import streamlit as st
from components.state import initialized_state
from components.chatbot import chatbot_preview_ui
from components.personalization import personalization_ui

# Set up the page layout
st.set_page_config(layout="wide")

# Initialize the session state
initialized_state()

# Create two columns for side-by-side layout
col1, col2 = st.columns(2)

with col1:
    # Personalization UI
    personalization_ui()

with col2:
    # Chatbot Preview UI
    chatbot_preview_ui()

# Generate dataset button
if st.button("Generate Dataset", key="generate_dataset"):
    # TODO: Implement this functionality
    # Call API for generating dataset

    # Display generating dataset status

    # If finished, add a success message and download the dataset
    pass
