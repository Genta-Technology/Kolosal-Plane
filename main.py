"""Main UI for Kolosal Plane"""

import streamlit as st

from interface.augmentation_interface import augmentation_interface
from interface.finetuning_interface import finetuning_interface

st.set_page_config(layout="wide")

st.title("Kolosal Plane Web Interface")

tabs = st.tabs(["Data Augmentation", "LLM Finetuning"])

with tabs[0]:
    augmentation_interface()

with tabs[1]:
    finetuning_interface()
