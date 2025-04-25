"""UI Kolosal Knowledge Augmentation"""
import streamlit as st

from interface.prompts_interface import knowledge_prompt_interface
from interface.configuration_interface import knowledge_configuration_interface
from interface.augmentation_interface import knowledge_augmentation_interface

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"

# The UI is seperated to 3 slides
# 1. Configure the augmentation prompts
# 2. Configure the augmentation parameters
# 3. Start the augmentation process


total_slides = 3


def augmentation_interface():
    if st.session_state.knowledge_slide == 1:
        knowledge_prompt_interface()

    elif st.session_state.knowledge_slide == 2:
        knowledge_configuration_interface()

    elif st.session_state.knowledge_slide == 3:
        knowledge_augmentation_interface()

    # Display navigation controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.session_state.knowledge_slide > 1:
            if st.button(label="← Previous",
                         key="prev_knowledge_slide"):
                st.session_state.knowledge_slide -= 1
                st.rerun()

    with col3:
        if st.session_state.knowledge_slide < total_slides:
            if st.button(label="Next →",
                         key="next_knowledge_slide"):
                st.session_state.knowledge_slide += 1
                st.rerun()

    # Display slide counter
    with col2:
        st.markdown(
            f"<div style='text-align: center;'>Slide {st.session_state.knowledge_slide} of {total_slides}</div>", unsafe_allow_html=True)
