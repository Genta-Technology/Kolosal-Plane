"""API Schema for Kolosal Plane"""
from typing import List, Optional, Dict, Literal, Any
from pydantic import BaseModel


class LLMConfiguration(BaseModel):
    """Configuration for Language Model"""
    model_provider: Literal["azure_openai", "openai",
                            "genta", "fireworks", "anthropic"]
    model_name: str
    model_parameters: Optional[Dict] = {}
    # API credentials
    api_key: str
    base_url: Optional[str] = None
    api_version: Optional[str] = None


class RequestEmbeddingAugmentation(BaseModel):
    """Request start augmentation of knowledge based dataset augmentation"""
    documents: List[str]
    instruction: str
    question_per_document: Optional[int] = 100
    batch_size: Optional[int] = 10
    llm_config: LLMConfiguration


class RequestKnowledgeAugmentation(BaseModel):
    """Request augmentation of knowledge based dataset"""
    documents: List[str]
    conversation_starter_instruction: str
    conversation_personalization_instruction: str
    system_prompt: str
    conversation_starter_count: Optional[int] = 10
    max_conversation_length: Optional[int] = 10
    batch_size: Optional[int] = 16
    llm_config: LLMConfiguration
    tlm_config: Optional[LLMConfiguration] = None


class ResponseAugmentation(BaseModel):
    """Response of knowledge based dataset augmentation"""
    generated_dataset: Dict
    metadata: Dict[str, int]


class AugmentationJobResponse(BaseModel):
    """Response for job submission"""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status"""
    job_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
