"""Document interface for Kolosal Plane"""
import streamlit as st
import pandas as pd


def documents_interface():
    """
    Creates a Streamlit interface for managing documents in the knowledge augmentation system.
    This function provides a UI for users to:
    - Upload documents from a CSV file (using the first column)
    - Manually edit documents in a data editor
    - Save document changes to the session state
    The documents are stored in the session state under 'knowledge_augmentation_config.documents'.
    Returns:
        None
    Notes:
        - CSV uploads will only use the first column of the file
        - The interface includes success/error messages for various operations
        - Changes are only persisted when the "Save Document" button is clicked
    """

    st.subheader("Documents")

    # Initialize the documents dataframe if it doesn't exist
    if "documents_df" not in st.session_state:
        current_documents = st.session_state.knowledge_augmentation_config.get(
            "documents", [])

        st.session_state.documents_df = pd.DataFrame(
            {"Documents": current_documents})
    
    if st.button("Load example documents", key="load_example_documents"):
        st.session_state.documents_df = pd.read_csv("example\documents.csv")

    uploaded_file = st.file_uploader(
        "Upload CSV (optional)",
        type=["csv"],
        help="Upload a CSV file. We'll use the first column and rename it to 'Documents'",
        key="csv_upload"
    )

    if st.button("Load CSV", key="load_csv_button"):
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if not uploaded_df.empty:
                    first_column = uploaded_df.iloc[:, 0]
                    st.session_state.documents_df = pd.DataFrame(
                        {"Documents": first_column})
                    st.success(
                        f"Loaded {len(st.session_state.documents_df)} documents from CSV")
                else:
                    st.error("Uploaded CSV is empty")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        else:
            st.error("Please upload a CSV file first")

    edited_df = st.data_editor(
        st.session_state.documents_df,
        num_rows="dynamic",
        column_config={"Documents": "Document"},
        height=400,
        key="documents_editor",
        use_container_width=True
    )

    if st.button("Save Document", key="save_document_button"):
        if "documents_editor" in st.session_state:
            # Update the main documents dataframe
            st.session_state.knowledge_augmentation_config["documents"] = edited_df["Documents"].to_list(
            )
            
            st.session_state.embeddings_augmentation_config["documents"] = edited_df["Documents"].to_list(
            )

            st.success("Document changes saved successfully!")
        else:
            st.error("No documents to save")
