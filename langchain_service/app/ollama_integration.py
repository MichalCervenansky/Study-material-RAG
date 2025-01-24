import json
import requests
from typing import Iterator, Optional, List, Dict


class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: str,
            stream: bool = True,
            format: Optional[dict] = None
    ) -> Iterator[str]:
        """
        Chat using the Ollama API.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            model (str): Model name to use
            stream (bool): Whether to stream the response
            format (dict, optional): JSON schema for structured output

        Yields:
            str: Generated response chunks
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if format:
            payload["format"] = format

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                stream=stream,
                timeout=(2, None)
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "message" in json_response:
                            yield json_response["message"]["content"]
            else:
                json_response = response.json()
                if "message" in json_response:
                    yield json_response["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
