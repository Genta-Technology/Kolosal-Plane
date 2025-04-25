"""Main UI interface for Kolosal Plane"""

import pandas as pd
import streamlit as st

from interface.knowledge_interface import augmentation_interface as knowledge_augmentation_interface
from interface.embedding_interface import augmentation_interface as embedding_augmentation_interface
from interface.general_interface import general_interface 
from interface.status_interface import augmentation_status_interface


st.set_page_config(layout="wide")

st.title("Kolosal Plane Interface")

# Load all session state
if "documents_df" not in st.session_state:
    st.session_state.documents_df = pd.DataFrame(columns=["Documents"])
    
if 'general_slide' not in st.session_state:
    st.session_state.general_slide = 1

if 'knowledge_slide' not in st.session_state:
    st.session_state.knowledge_slide = 1

if 'embedding_slide' not in st.session_state:
    st.session_state.embedding_slide = 1

if "knowledge_augmentation_config" not in st.session_state:
    st.session_state.knowledge_augmentation_config = {}
    # Consist of:
    # 1. documents
    # 2. conversation_starter_instruction
    # 3. conversation_personalization_instruction
    # 4. system_prompt
    # 5. llm_config
    # 6. tlm_config
    # 7. conversation_starter_count
    # 8. max_conversation_length
    # 9. batch_size

if "embeddings_augmentation_config" not in st.session_state:
    st.session_state.embeddings_augmentation_config = {}
    # Consist of:
    # 1. documents
    # 2. instruction
    # 3. question_per_document
    # 4. batch_size
    # 5. llm_config

if "job_id" not in st.session_state:
    st.session_state.job_id = None

tabs = st.tabs(["General Configuration", "Knowledge Augmentation",
               "Embeddings Augmentation", "Augmentation Status"])
with tabs[0]:
    general_interface()

with tabs[1]:
    knowledge_augmentation_interface()

with tabs[2]:
    embedding_augmentation_interface()

with tabs[3]:
    augmentation_status_interface()