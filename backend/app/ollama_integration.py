import aiohttp
import json
from typing import AsyncGenerator
from .logger_config import get_logger, log_time

logger = get_logger(__name__)


class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.timeout = aiohttp.ClientTimeout(total=3600)
        logger.info(f"Initialized OllamaAPI with base URL: {base_url}")

    @log_time(logger)
    async def chat(
            self,
            messages: list[dict[str, str]],
            model: str,
            format: dict | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat using Ollama API
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        if format:
            payload["format"] = format

        try:
            logger.info(f"Starting async chat request with model: {model}")
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                        self.chat_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        if line:
                            json_response = json.loads(line)
                            if "message" in json_response:
                                yield json_response["message"]["content"]

            logger.info("Finished streaming chat response")

        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise
