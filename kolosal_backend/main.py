"""
API for Kolosal backend service, using FastAPI to handle data augmentation process and prompt generation request
"""
import os
import uuid
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header

from kolosal_backend.api.schemas import PersonalizationRequest, KnowledgeRequest, JobStatusRequest, JobResultRequest, JobCreateResponse, JobStatusResponse, JobResultResponse
from kolosal_backend.api.models import get_llm, get_slm

from kolosal_backend.pipeline.parameter import PersonalizationParameter, KnowledgeParameter
from kolosal_backend.pipeline.personalization_pipeline import personalization_pipeline
from kolosal_backend.pipeline.knowledge_pipeline import knowledge_pipeline

# Initializing the LLM and SLM models
llm = get_llm()
slm = get_slm()

# API key for authorization on Kolosal, modifiedable from env
KOLOSAL_API_KEY = os.getenv("KOLOSAL_API_KEY")

# In-memory storage for job status and results
jobs: Dict[str, Dict] = {}

app = FastAPI()


@app.post("/personalization")
async def submit_personalization_job(request: PersonalizationRequest,
                                     background_tasks: BackgroundTasks,
                                     authorization: str = Header(None)):
    """
    Submits a personalization job to be processed in the background.
    Args:
        request (PersonalizationRequest): The request object containing personalization parameters.
        background_tasks (BackgroundTasks): The background tasks manager to handle asynchronous tasks.
        authorization (str, optional): The API key for authorization. Defaults to Header(None).
    Raises:
        HTTPException: If the provided API key is invalid.
    Returns:
        dict: A dictionary containing the job ID and the status of the job submission.
    """

    # Authorization
    if authorization != KOLOSAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "in progress", "result": None}

    # Add the job to background tasks with the provided request data
    background_tasks.add_task(personalization_pipeline, job_id, PersonalizationParameter(
        conversation_starter_instruction=request.conversation_starter_instruction,
        conversation_personalization_instruction=request.conversation_personalization_instruction,
        max_conversations=request.max_conversations,
        conversation_starter_count=request.conversation_starter_count,
        llm_model=llm,
        slm_model=slm
    ))

    return JobCreateResponse(job_id=job_id, status="Job submitted")


@app.post("/knowledge")
async def submit_knowledge_job(request: KnowledgeRequest,
                               background_tasks: BackgroundTasks,
                               authorization: str = Header(None)):
    """
    Submits a knowledge augmentation job to be processed in the background.
    Args:
        request (KnowledgeRequest): The request object containing knowledge augmentation parameters.
        background_tasks (BackgroundTasks): The background tasks manager to handle asynchronous tasks.
        authorization (str, optional): The API key for authorization. Defaults to Header(None).
    Raises:
        HTTPException: If the provided API key is invalid.
    Returns:
        dict: A dictionary containing the job ID and the status of the job submission.
    """

    # Authorization
    if authorization != KOLOSAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "in progress", "result": None}

    # Add the job to background tasks with the provided request data
    background_tasks.add_task(knowledge_pipeline, job_id, KnowledgeParameter(
        documents=request.documents,
        conversation_starter_instruction=request.conversation_starter_instruction,
        conversation_personalization_instruction=request.conversation_personalization_instruction,
        max_conversations=request.max_conversations,
        conversation_starter_count=request.conversation_starter_count,
        llm_model=llm,
        slm_model=slm
    ))

    return JobCreateResponse(job_id=job_id, status="Job submitted")


@app.post("/job_status")
async def status_job(request: JobStatusRequest, authorization: str = Header(None)):
    """
    Check the status of a job.
    Args:
        request (JobStatusRequest): The request object containing the job ID.
        authorization (str, optional): The API key for authorization. Defaults to Header(None).
    Raises:
        HTTPException: If the API key is invalid (status code 401).
        HTTPException: If the job is not found or already retrieved (status code 404).
    Returns:
        JobStatusResponse: The response object containing the job status.
    """

    # Authorization
    if authorization != KOLOSAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    job = jobs.get(request.job_id)
    if not job:
        raise HTTPException(
            status_code=404, detail="Job not found or already retrieved")

    if job['status'] == 'in progress':
        return JobStatusResponse(ready=False, status="Job in progress")

    return JobStatusResponse(ready=True, status="completed")


@app.post("/job_result")
async def result_job(request: JobResultRequest, authorization: str = Header(None)):
    """
    Retrieve the result of a job.
    Args:
        request (JobResultRequest): The request object containing the job ID.
        authorization (str, optional): The API key for authorization. Defaults to Header(None).
    Raises:
        HTTPException: If the API key is invalid (status code 401).
        HTTPException: If the job is not found or already retrieved (status code 404).
    Returns:
        JobResultResponse: The response object containing the job result.
    """

    # Authorization
    if authorization != KOLOSAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    job = jobs.get(request.job_id)
    if not job:
        raise HTTPException(
            status_code=404, detail="Job not found or already retrieved")

    if job['status'] == 'in progress':
        raise HTTPException(status_code=404, detail="Job in progress")

    return JobResultResponse(dataset=job['result'])
