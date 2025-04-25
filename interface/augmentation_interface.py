"""Augmentation Interface for Kolosal Plane"""

import streamlit as st

from api.api_helper import start_knowledge_augmentation, start_embedding_augmentation


def knowledge_augmentation_interface():
    """
    Renders the knowledge augmentation interface in the Streamlit application.
    This function provides UI components to:
    1. Display the current knowledge augmentation configuration as JSON
    2. Validate the configuration payload
    3. Start the knowledge augmentation process
    The interface includes:
    - A display of the current configuration stored in session state
    - A "Validate Payload" button to check if the configuration is valid
    - A "Start Augmentation" button to initiate the knowledge augmentation process
    When augmentation is started, it sends the configuration to the backend API
    and stores the resulting job ID in the session state for tracking.
    Returns:
        None
    Side effects:
        - Updates st.session_state.job_id when augmentation is started
        - Displays success/error messages through the Streamlit interface
    """

    # Display current configuration
    st.subheader("Current Configuration")
    st.json(st.session_state.knowledge_augmentation_config)

    # Validate Payload
    if st.button("Validate Payload"):
        if validate_knowledge_payload(st.session_state.knowledge_augmentation_config):
            st.success("Payload is valid")

    # Start Augmentation
    if st.button("Start Augmentation"):
        # Check if the configuration is complete and only one job id is present
        if validate_knowledge_payload(st.session_state.knowledge_augmentation_config) and st.session_state.job_id is None:
            # Prepare the payload for the API request
            payload = st.session_state.knowledge_augmentation_config.copy()

            # Request the API
            response = start_knowledge_augmentation(
                documents=payload["documents"],
                conversation_starter_instruction=payload["conversation_starter_instruction"],
                conversation_personalization_instruction=payload[
                    "conversation_personalization_instruction"],
                system_prompt=payload["system_prompt"],
                llm_config=payload["llm_config"],
                tlm_config=payload.get("tlm_config"),
                conversation_starter_count=payload["conversation_starter_count"],
                max_conversation_length=payload["max_conversation_length"],
                batch_size=payload["batch_size"]
            )

            st.session_state.job_id = response["job_id"]

            # Display the job id
            st.success(
                f"Message: {response['message']}, Job ID: {response['job_id']}, Status: {response['status']}")


def embedding_augmentation_interface():
    """
    Creates a Streamlit interface for embedding augmentation.
    This function displays the current embedding augmentation configuration,
    provides buttons to validate the configuration payload and start the augmentation process.
    When the "Start Augmentation" button is clicked, it validates the configuration,
    checks that no job is currently running, and then sends the augmentation 
    request to the API with the configured parameters.
    The function updates the session state with the job_id from the API response
    and displays success messages with job information.
    Returns:
        None
    Session State Variables:
        embeddings_augmentation_config (dict): Configuration parameters for the embedding augmentation
        job_id (str): ID of the current augmentation job, if one is running
    """

    # Display current configuration
    st.subheader("Current Configuration")
    st.json(st.session_state.embeddings_augmentation_config)

    # Validate Payload
    if st.button("Validate Payload", key="validate_embedding_button"):
        if validate_embedding_payload(st.session_state.embeddings_augmentation_config):
            st.success("Payload is valid")

    # Start Augmentation
    if st.button("Start Augmentation", key="start_embedding_button"):
        # Check if the configuration is complete and only one job id is present
        if validate_embedding_payload(st.session_state.embeddings_augmentation_config) and st.session_state.job_id is None:
            # Prepare the payload for the API request
            payload = st.session_state.embeddings_augmentation_config.copy()

            # Request the API
            response = start_embedding_augmentation(
                documents=payload["documents"],
                instruction=payload["instruction"],
                question_per_document=payload["question_per_document"],
                llm_config=payload["llm_config"],
                batch_size=payload["batch_size"]
            )

            st.session_state.job_id = response["job_id"]

            # Display the job id
            st.success(
                f"Message: {response['message']}, Job ID: {response['job_id']}, Status: {response['status']}")


def validate_knowledge_payload(payload: dict) -> bool:
    """
    Validates if the payload is complete according to requirements.

    Requirements:
    - documents should have at least one element (string) inside
    - tlm_config could be None
    - llm_config should not be None
    - All other fields should not be None or empty strings

    Args:
        payload (dict): The payload to validate

    Returns:
        bool: True if payload is valid, False otherwise
    """
    errors = []

    # Check if documents has at least one element
    if not payload.get("documents") or not isinstance(payload["documents"], list) or len(payload["documents"]) == 0:
        errors.append("Documents must contain at least one element")
    elif not all(isinstance(doc, str) for doc in payload["documents"]):
        errors.append("All documents must be strings")

    # Check if llm_config is not None
    if payload.get("llm_config") is None:
        errors.append("llm_config cannot be None")

    # tlm_config can be None, so no check needed for that

    # Check other fields are not None or empty strings
    required_fields = [
        "conversation_starter_instruction",
        "conversation_personalization_instruction",
        "system_prompt",
        "conversation_starter_count",
        "max_conversation_length",
        "batch_size"
    ]

    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
        elif payload[field] is None:
            errors.append(f"{field} cannot be None")
        elif isinstance(payload[field], str) and not payload[field].strip():
            errors.append(f"{field} cannot be an empty string")

    # Display warnings for any errors
    if errors:
        for error in errors:
            st.warning(error)
        return False

    return True


def validate_embedding_payload(payload: dict) -> bool:
    """
    Validates if the payload is complete according to requirements.

    Requirements:
    - documents should have at least one element (string) inside
    - tlm_config could be None
    - llm_config should not be None
    - All other fields should not be None or empty strings

    Args:
        payload (dict): The payload to validate

    Returns:
        bool: True if payload is valid, False otherwise
    """
    errors = []

    # Check if documents has at least one element
    if not payload.get("documents") or not isinstance(payload["documents"], list) or len(payload["documents"]) == 0:
        errors.append("Documents must contain at least one element")
    elif not all(isinstance(doc, str) for doc in payload["documents"]):
        errors.append("All documents must be strings")

    # Check if llm_config is not None
    if payload.get("llm_config") is None:
        errors.append("llm_config cannot be None")

    # tlm_config can be None, so no check needed for that

    # Check other fields are not None or empty strings
    required_fields = [
        "instruction",
        "question_per_document",
        "batch_size"
    ]

    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
        elif payload[field] is None:
            errors.append(f"{field} cannot be None")
        elif isinstance(payload[field], str) and not payload[field].strip():
            errors.append(f"{field} cannot be an empty string")

    # Display warnings for any errors
    if errors:
        for error in errors:
            st.warning(error)
        return False

    return True
