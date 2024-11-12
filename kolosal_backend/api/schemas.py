"""Pydantic models for Kolosal API request and response schemas."""
from pydantic import BaseModel
from typing import List, Dict


class PersonalizationRequest(BaseModel):
    """
    PersonalizationRequest schema for handling personalization requests.
    Attributes:
        conversation_starter_instruction (str): Instruction for starting a conversation.
        conversation_personalization_instruction (str): Instruction for personalizing a conversation.
        max_conversations (int): Maximum number of conversations allowed.
        conversation_starter_count (int): Number of conversation starters to generate.
    """

    conversation_starter_instruction: str
    conversation_personalization_instruction: str
    max_conversations: int
    conversation_starter_count: int


class KnowledgeRequest(BaseModel):
    """
    KnowledgeRequest schema for handling knowledge request data.
    Attributes:
        documents (List[str]): A list of document strings.
        conversation_starter_instruction (str): Instruction for starting a conversation.
        conversation_personalization_instruction (str): Instruction for personalizing a conversation.
        max_conversations (int): Maximum number of conversations allowed.
        conversation_starter_count (int): Number of conversation starters.
    """

    documents: List[str]
    conversation_starter_instruction: str
    conversation_personalization_instruction: str
    max_conversations: int
    conversation_starter_count: int


class JobCreateResponse(BaseModel):
    """
    JobCreateResponse is a Pydantic model representing the response for a job creation request.
    Attributes:
        task_id (str): The unique identifier for the created task.
        status (str): The current status of the created task.
    """

    job_id: str
    status: str


class JobStatusRequest(BaseModel):
    """
    JobStatusRequest represents the request schema for job status.
    Attributes:
        job_id (str): The unique identifier for the job.
    """

    job_id: str


class JobStatusResponse(BaseModel):
    """
    JobStatusResponse represents the response schema for job status.
    Attributes:
        ready (bool): Indicates whether the job is ready.
        status (str): The current status of the job.
    """

    ready: bool
    status: str


class JobResultRequest(BaseModel):
    """
    JobResultRequest represents the request schema for a job result.
    Attributes:
        job_id (str): The unique identifier for the job.
    """

    job_id: str


class JobResultResponse(BaseModel):
    """
    JobResultResponse represents the response schema for a job result.
    Attributes:
        dataset (List[Dict[str, str]]): A list of dictionaries where each dictionary represents a dataset with string keys and string values.
    """

    dataset: List[Dict[str, str]]
