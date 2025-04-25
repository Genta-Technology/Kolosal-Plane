"""API For Kolosal Plane"""
import uuid
from typing import Dict, Any
import polars as pl
from fastapi import FastAPI, HTTPException, BackgroundTasks

from kolosal_plane.augmentations.knowledge_simplified import AsyncSimpleKnowledge
from kolosal_plane.augmentations.embeddings import AsyncEmbeddingAugmentation
from kolosal_plane.utils.llm import create_llm_from_config
from api.schema import RequestEmbeddingAugmentation, RequestKnowledgeAugmentation, ResponseAugmentation, AugmentationJobResponse, JobStatusResponse

# Job storage
kolosal_jobs: Dict[str, Any] = {}

app = FastAPI(title="Kolosal Plane API",
              description="API for dataset augmentation in Kolosal Plane",
              version="1.0.0")


# Utility function to convert polars DataFrame to dict for the API response
def convert_df_to_dict(df: pl.DataFrame) -> Dict:
    """Convert a polars DataFrame to a dictionary format suitable for API response"""
    if "chat_history" in df.columns:
        # For knowledge augmentation
        return {
            "rows": df.to_dicts()
        }
    else:
        # For embedding augmentation
        return {
            "rows": df.to_dicts()
        }


@app.post("/embedding-augmentation", response_model=AugmentationJobResponse)
async def start_embedding_augmentation(request: RequestEmbeddingAugmentation):
    """Start an embedding augmentation job"""
    job_id = str(uuid.uuid4())

    # Create LLM instance from config
    llm = create_llm_from_config(request.llm_config)

    # Create augmentation instance
    augmentation = AsyncEmbeddingAugmentation(
        documents=request.documents,
        instruction=request.instruction,
        lm=llm,
        question_per_document=request.question_per_document,
        batch_size=request.batch_size
    )

    # Start the augmentation task
    _task = augmentation.start_augmentation()

    # Store the job
    kolosal_jobs[job_id] = {
        "type": "embedding",
        "augmentation": augmentation,
    }

    return AugmentationJobResponse(
        job_id=job_id,
        status="running",
        message="Embedding augmentation job started successfully"
    )


@app.post("/knowledge-augmentation", response_model=AugmentationJobResponse)
async def start_knowledge_augmentation(request: RequestKnowledgeAugmentation,
                                       background_tasks: BackgroundTasks):
    """Start a knowledge augmentation job"""
    job_id = str(uuid.uuid4())

    # Create LLM instances from config
    llm = create_llm_from_config(request.llm_config)
    tlm = None
    if request.tlm_config:
        tlm = create_llm_from_config(request.tlm_config)

    # Create augmentation instance
    augmentation = AsyncSimpleKnowledge(
        documents=request.documents,
        conversation_starter_instruction=request.conversation_starter_instruction,
        conversation_personalization_instruction=request.conversation_personalization_instruction,
        system_prompt=request.system_prompt,
        max_conversations=request.max_conversation_length,
        conversation_starter_count=request.conversation_starter_count,
        batch_size=request.batch_size,
        llm_model=llm,
        thinking_model=tlm
    )

    # Start the augmentation task
    _task = augmentation.start_augmentation()

    # Store the job
    kolosal_jobs[job_id] = {
        "type": "knowledge",
        "augmenter": augmentation
    }

    return AugmentationJobResponse(
        job_id=job_id,
        status="running",
        message="Knowledge augmentation job started successfully"
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in kolosal_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = kolosal_jobs[job_id]
    augmenter = job["augmenter"]

    status, metadata = augmenter.get_status()

    # Get current results to calculate progress
    df, _ = augmenter.get_result()

    progress = None
    if job["type"] == "knowledge":
        total_rows = len(df)
        completed_rows = len(df.filter(pl.col("response") != ""))
        progress = {
            "total_rows": total_rows,
            "completed_rows": completed_rows,
            "percentage": int((completed_rows / max(1, total_rows)) * 100)
        }
    elif job["type"] == "embedding":
        progress = {
            "total_documents": len(augmenter.documents),
            "questions_generated": len(df),
            "questions_per_doc_target": augmenter.question_per_document,
            "total_questions_target": len(augmenter.documents) * augmenter.question_per_document,
            "percentage": int((len(df) / max(1, len(augmenter.documents) * augmenter.question_per_document)) * 100)
        }

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        metadata=metadata
    )


@app.get("/jobs/{job_id}/result", response_model=ResponseAugmentation)
async def get_job_result(job_id: str):
    """Get the current result of a job"""
    if job_id not in kolosal_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = kolosal_jobs[job_id]
    augmenter = job["augmenter"]

    df, metadata = augmenter.get_result()

    return ResponseAugmentation(
        generated_dataset=convert_df_to_dict(df),
        metadata=metadata
    )


@app.delete("/jobs/{job_id}", response_model=AugmentationJobResponse)
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in kolosal_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = kolosal_jobs[job_id]
    augmenter = job["augmenter"]

    # Cancel the job
    augmenter.cancel_augmentation()

    # Get final results
    _, _ = augmenter.get_result()

    # Clean up (optional - you might want to keep it for a while)
    # active_jobs.pop(job_id)

    return AugmentationJobResponse(
        job_id=job_id,
        status="cancelled",
        message="Job cancelled successfully"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
