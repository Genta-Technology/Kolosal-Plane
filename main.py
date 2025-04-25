"""Run both Kolosal Plane Streamlit UI and API apps in parallel using threading."""
import subprocess
import threading
import sys


def run_streamlit():
    """Run Streamlit app"""
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                   "interface.py", "--server.port", "8501"])


def run_fastapi():
    """Run FastAPI app"""
    subprocess.run([sys.executable, "-m", "uvicorn",
                   "api.api:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    # Create threads for both apps
    streamlit_thread = threading.Thread(target=run_streamlit)
    fastapi_thread = threading.Thread(target=run_fastapi)

    # Start both threads
    streamlit_thread.start()
    fastapi_thread.start()

    # Wait for both threads to complete
    streamlit_thread.join()
    fastapi_thread.join()
