"""UI Kolosal Knowledge Augmentation"""
import streamlit as st
import pandas as pd

from interface.model_config import model_config_interface
from interface.documents_interface import documents_interface
from interface.prompts_interface import knowledge_prompt_interface
from interface.configuration_interface import knowledge_configuration_interface
from interface.augmentation_interface import knowledge_augmentation_interface

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"

# The UI is seperated to 6 slides
# 1. Setup the LLM provider and parameters
# 2. Upload the documents to be augmented
# 3. Configure the augmentation prompts
# 4. Configure the augmentation parameters
# 5. Start the augmentation process
# 6. Monitor the augmentation


total_slides = 6


def augmentation_interface():
    if st.session_state.knowledge_slide == 1:
        model_config_interface()

    elif st.session_state.knowledge_slide == 2:
        documents_interface()

    elif st.session_state.knowledge_slide == 3:
        knowledge_prompt_interface()
        
    elif st.session_state.knowledge_slide == 4:
        knowledge_configuration_interface()
    
    elif st.session_state.knowledge_slide == 5:
        knowledge_augmentation_interface()

    # Display navigation controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.session_state.knowledge_slide > 1:
            if st.button("← Previous"):
                st.session_state.knowledge_slide -= 1
                st.rerun()

    with col3:
        if st.session_state.knowledge_slide < total_slides:
            if st.button("Next →"):
                st.session_state.knowledge_slide += 1
                st.rerun()

    # Display slide counter
    with col2:
        st.markdown(
            f"<div style='text-align: center;'>Slide {st.session_state.knowledge_slide} of {total_slides}</div>", unsafe_allow_html=True)
