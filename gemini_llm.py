from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr
import os
import google.generativeai as genai

class GeminiLLM(LLM):
    model: str = 'gemini-1.5-flash'
    temperature: float = 0.7

    _client: genai.GenerativeModel = PrivateAttr()

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self.model)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.generate_content(prompt)
        return response.text
