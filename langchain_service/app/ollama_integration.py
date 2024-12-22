import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the Ollama API and get a response.

    Args:
        prompt (str): The input prompt.
        model (str): The model to use for generation. Defaults to DEFAULT_MODEL.

    Returns:
        str: The response from the Ollama API.
    """
    payload = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
