import os
import requests
from typing import Dict

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = "https://api.ollama.ai/v1/llm"

def query_ollama(prompt: str) -> str:
    """
    Send a prompt to the Ollama API and get a response.

    Args:
        prompt (str): The input prompt.

    Returns:
        str: The response from the Ollama API.
    """
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    payload = {"prompt": prompt}
    response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
