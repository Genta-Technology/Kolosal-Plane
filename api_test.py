# Simple test code for Kolosal Plane API
import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if you have one)
load_dotenv()

# Base URL - update this to match your API deployment
BASE_URL = "http://localhost:8000"  # or your actual API URL

# Function to print responses in a readable format


def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print("-" * 50)


# LLM configuration using Azure OpenAI - similar to your example
llm_config = {
    "model_provider": "azure_openai",
    "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION"),
    "model_name": "gpt-4o",
    "model_parameters": {
        "max_new_tokens": 1024
    }
}

# 1. Test health check endpoint
print("Testing health check endpoint...")
response = requests.get(f"{BASE_URL}/health")
print_response(response)

# 2. Test embedding augmentation endpoint
print("\nTesting embedding augmentation endpoint...")
embedding_request = {
    "documents": [
        "Kolosal Plane is a data augmentation system.",
        "The system can generate questions and conversations."
    ],
    "instruction": "Generate questions that someone might ask about these documents.",
    "llm_config": llm_config,
    "question_per_document": 10,
    "batch_size": 64
}

response = requests.post(
    f"{BASE_URL}/embedding-augmentation",
    json=embedding_request
)
print_response(response)

# Save job ID for later use
if response.status_code == 200:
    embedding_job_id = response.json().get("job_id")
    print(f"Saved embedding job ID: {embedding_job_id}")
else:
    embedding_job_id = None

# 3. Test knowledge augmentation endpoint
print("\nTesting knowledge augmentation endpoint...")
knowledge_request = {
    "documents": [
        "Kolosal Plane is a data augmentation system.",
        "The system can generate questions and conversations."
    ],
    "conversation_starter_instruction": "Start an interesting conversation about this document.",
    "conversation_personalization_instruction": "Make the conversation relevant to data scientists.",
    "system_prompt": "You are a helpful AI assistant.",
    "max_conversation_length": 3,
    "conversation_starter_count": 3,
    "batch_size": 16,
    "llm_config": llm_config
}

response = requests.post(
    f"{BASE_URL}/knowledge-augmentation",
    json=knowledge_request
)
print_response(response)

# Save job ID for later use
if response.status_code == 200:
    knowledge_job_id = response.json().get("job_id")
    print(f"Saved knowledge job ID: {knowledge_job_id}")
else:
    knowledge_job_id = None

# Wait a moment for jobs to start processing
print("\nWaiting for jobs to start processing...")
time.sleep(2)

# 4. Test job status endpoint for embedding job
if embedding_job_id:
    print(f"\nChecking status of embedding job {embedding_job_id}...")
    response = requests.get(f"{BASE_URL}/jobs/{embedding_job_id}/status")
    print_response(response)

# 5. Test job status endpoint for knowledge job
if knowledge_job_id:
    print(f"\nChecking status of knowledge job {knowledge_job_id}...")
    response = requests.get(f"{BASE_URL}/jobs/{knowledge_job_id}/status")
    print_response(response)

# 6. Test job result endpoint for embedding job
if embedding_job_id:
    print(f"\nChecking result of embedding job {embedding_job_id}...")
    response = requests.get(f"{BASE_URL}/jobs/{embedding_job_id}/result")
    print_response(response)

# 7. Test job result endpoint for knowledge job
if knowledge_job_id:
    print(f"\nChecking result of knowledge job {knowledge_job_id}...")
    response = requests.get(f"{BASE_URL}/jobs/{knowledge_job_id}/result")
    print_response(response)

# 8. Test cancelling a job
if embedding_job_id:
    print(f"\nCancelling embedding job {embedding_job_id}...")
    response = requests.delete(f"{BASE_URL}/jobs/{embedding_job_id}")
    print_response(response)

# 9. Test job not found scenario
print("\nTesting job not found scenario...")
response = requests.get(f"{BASE_URL}/jobs/non-existent-job-id/status")
print_response(response)

print("API testing completed!")
