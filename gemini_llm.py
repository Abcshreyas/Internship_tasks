from langchain.llms.base import LLM
from typing import Optional, List
import os 
import google.generativeai as genai

class GeminiLLM(LLM):
    model: str = 'gemini-pro'
    temperature: float = 0.7

def __init__(self, api_key: str, **kwargs):
    super().__init__(**kwargs)
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    self.client = genai.GenerativeModel(self.model)

@property
def _llm_type(self) -> str:
    return "gemini"

def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    respnse = self.client.generate_content(prompt)
    return response.text