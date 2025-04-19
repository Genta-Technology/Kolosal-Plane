"""Augmentation Status Interface"""

import streamlit as st
import json
from api.api_helper import get_job_status, get_job_result, cancel_job, health_check


def knowledge_augmentation_status_interface():
    # Add job ID manually
    st.subheader("Job Management")
    manual_job_id = st.text_input(
        "Enter Job ID manually:", key="manual_job_id")

    if st.button("Set Job ID"):
        if manual_job_id:
            st.session_state.job_id = manual_job_id
            st.success(f"Job ID set to: {manual_job_id}")
        else:
            st.error("Please enter a valid Job ID")

    # Display current job ID
    job_id = st.session_state.get("job_id")
    if job_id:
        st.info(f"Current Job ID: {job_id}")
    else:
        st.warning(
            "No job ID found. Please start a job or enter a job ID manually.")
        return
    

    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Job Status", "Job Result", "Job Control", "API Health"])

    # Tab 1: Job Status
    with tab1:
        st.subheader("Job Status")
        if st.button("Check Status", key="check_status"):
            try:
                status_data = get_job_status(job_id)
                st.success(f"Status: {status_data['status']}")

                # Display progress information
                if status_data.get("progress"):
                    st.subheader("Progress")
                    progress = status_data["progress"]

                    # Create a progress bar if percentage is available
                    if "percentage" in progress:
                        st.progress(progress["percentage"] / 100)
                        st.write(f"Completion: {progress['percentage']}%")

                    # Display all progress details
                    for key, value in progress.items():
                        if key != "percentage":
                            st.write(
                                f"{key.replace('_', ' ').title()}: {value}")

                # Display metadata
                if status_data.get("metadata"):
                    st.subheader("Metadata")
                    for key, value in status_data["metadata"].items():
                        st.write(f"{key.replace('_', ' ').title()}: {value}")

            except Exception as e:
                st.error(f"Error retrieving job status: {str(e)}")

    # Tab 2: Job Result
    with tab2:
        st.subheader("Job Result")
        if st.button("Get Results", key="get_results"):
            try:
                result_data = get_job_result(job_id)

                # Display dataset summary
                st.subheader("Generated Dataset")
                if "generated_dataset" in result_data and "rows" in result_data["generated_dataset"]:
                    rows = result_data["generated_dataset"]["rows"]
                    st.write(f"Total rows: {len(rows)}")

                    # Show preview of first few rows
                    if rows:
                        st.subheader("Preview (First 5 Rows)")
                        st.dataframe(rows[:5])

                    # JSON view toggle
                    if st.checkbox("View Full JSON"):
                        st.json(result_data)
                else:
                    st.warning("No dataset rows found in the result")

                # Display metadata
                if "metadata" in result_data:
                    st.subheader("Metadata")
                    for key, value in result_data["metadata"].items():
                        st.write(f"{key.replace('_', ' ').title()}: {value}")

            except Exception as e:
                st.error(f"Error retrieving job result: {str(e)}")

    # Tab 3: Job Control
    with tab3:
        st.subheader("Cancel Job")
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("Cancel Job", key="cancel_job"):
                try:
                    cancel_response = cancel_job(job_id)
                    st.success(
                        f"Job cancelled successfully. Status: {cancel_response['status']}")
                    st.write(f"Message: {cancel_response['message']}")
                except Exception as e:
                    st.error(f"Error cancelling job: {str(e)}")

        with col2:
            st.warning("⚠️ This action will stop the job and cannot be undone.")

    # Tab 4: API Health
    with tab4:
        st.subheader("API Health Check")
        if st.button("Check API Health", key="check_health"):
            try:
                health_response = health_check()
                if health_response.get("status") == "ok":
                    st.success("API is operational ✅")
                else:
                    st.error("API health check returned unexpected status")
            except Exception as e:
                st.error(f"API health check failed: {str(e)}")
