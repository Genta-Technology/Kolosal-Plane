"""API helper functions for embedding and knowledge augmentation."""

import requests
from typing import List, Dict, Any, Optional

# === Configuration ===
BASE_URL = "http://127.0.0.1:8000"     # Change to your API's host:port or domain

HEADERS = {"Content-Type": "application/json"}


def start_embedding_augmentation(
    documents: List[str],
    instruction_positive: str,
    instruction_negative: str,
    llm_config: Dict[str, Any],
    question_per_document: int = 100,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """
    Kick off an embedding augmentation job.
    """
    url: str = f"{BASE_URL}/embedding-augmentation"
    payload: Dict[str, Any] = {
        "documents": documents,
        "instruction_positive": instruction_positive,
        "instruction_negative": instruction_negative,
        "question_per_document": question_per_document,
        "batch_size": batch_size,
        "llm_config": llm_config,
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()


def start_knowledge_augmentation(
    documents: List[str],
    conversation_starter_instruction: str,
    conversation_personalization_instruction: str,
    system_prompt: str,
    llm_config: Dict[str, Any],
    tlm_config: Optional[Dict[str, Any]] = None,
    conversation_starter_count: int = 10,
    max_conversation_length: int = 10,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Kick off a knowledge augmentation job.
    """
    url: str = f"{BASE_URL}/knowledge-augmentation"
    payload: Dict[str, Any] = {
        "documents": documents,
        "conversation_starter_instruction": conversation_starter_instruction,
        "conversation_personalization_instruction": conversation_personalization_instruction,
        "system_prompt": system_prompt,
        "conversation_starter_count": conversation_starter_count,
        "max_conversation_length": max_conversation_length,
        "batch_size": batch_size,
        "llm_config": llm_config,
        "tlm_config": tlm_config,
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Poll the status of a running job.
    """
    url: str = f"{BASE_URL}/jobs/{job_id}/status"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_job_result(job_id: str) -> Dict[str, Any]:
    """
    Retrieve the current results of a job.
    """
    url: str = f"{BASE_URL}/jobs/{job_id}/result"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a running job.
    """
    url: str = f"{BASE_URL}/jobs/{job_id}"
    resp = requests.delete(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def health_check() -> Dict[str, str]:
    """
    Simple health check.
    """
    url: str = f"{BASE_URL}/health"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()
