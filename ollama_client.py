import requests
import json
from typing import List, Dict, Optional, Generator


class OllamaChatClient:
    """
    A reusable chat client for working with a local Ollama server.
    Supports system/user prompts, streaming, and conversation history.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.messages: List[Dict[str, str]] = []

        # Add optional system prompt at initialization
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant reply to the history."""
        self.messages.append({"role": "assistant", "content": content})

    def chat(self, user_message: str, stream: bool = False) -> str:
        """
        Send a user message and get a response (streaming or non-streaming).
        Returns the full reply as a string.
        """

        self.add_user_message(user_message)

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
        }

        url = f"{self.host}/api/chat"

        if not stream:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()

            reply = resp.json()["message"]["content"]
            self.add_assistant_message(reply)
            return reply

        # Streaming version
        reply_text = ""
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    delta = data.get("message", {}).get("content", "")
                    print(delta, end="", flush=True)
                    reply_text += delta

        print()  # newline after streaming output
        self.add_assistant_message(reply_text)
        return reply_text

    def reset(self):
        """Clear conversation history (except system prompt if present)."""
        system = None
        if self.messages and self.messages[0]["role"] == "system":
            system = self.messages[0]
        self.messages = []
        if system:
            self.messages.append(system)
