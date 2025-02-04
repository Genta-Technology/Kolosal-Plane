"""Main UI for Kolosal Plane"""
import pandas as pd
import streamlit as st

from interface.augmentation_interface import augmentation_interface
from interface.finetuning_interface import finetuning_interface

st.set_page_config(layout="wide")

st.title("Kolosal Plane Web Interface")

# Load all session state
if "llm" not in st.session_state:
    st.session_state.llm = None
if "slm" not in st.session_state:
    st.session_state.slm = None
if "thinking" not in st.session_state:
    st.session_state.slm = None
if "documents_df" not in st.session_state:
    st.session_state.documents_df = pd.DataFrame(columns=["Documents"])

tabs = st.tabs(["Data Augmentation", "LLM Finetuning"])

with tabs[0]:
    augmentation_interface()

with tabs[1]:
    finetuning_interface()
