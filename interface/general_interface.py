"""UI Kolosal General Setting"""
import streamlit as st

from interface.model_config import model_config_interface
from interface.documents_interface import documents_interface

# The UI is seperated to 2 slides
# 1. Configure the LLM
# 2. Configure the Documents

total_slides = 3


def general_interface():
    if st.session_state.general_slide == 1:
        model_config_interface()

    elif st.session_state.general_slide == 2:
        documents_interface()

    # Display navigation controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.session_state.general_slide > 1:
            if st.button("← Previous"):
                st.session_state.general_slide -= 1
                st.rerun()

    with col3:
        if st.session_state.general_slide < total_slides:
            if st.button("Next →"):
                st.session_state.general_slide += 1
                st.rerun()

    # Display slide counter
    with col2:
        st.markdown(
            f"<div style='text-align: center;'>Slide {st.session_state.general_slide} of {total_slides}</div>", unsafe_allow_html=True)
