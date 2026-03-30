"""
Gemini LLM Wrapper — provides a llama_cpp-compatible interface
so that chat.py works without modifications.

Usage:
    llm = GeminiLLM(model_name="gemini-2.0-flash")
    response = llm(prompt, max_tokens=150, temperature=0.0)
    # response == {"choices": [{"text": "..."}]}
"""

import os
import google.generativeai as genai  # type: ignore


class GeminiLLM:
    """
    Wraps the Google Gemini API behind the same callable interface
    that llama_cpp.Llama provides:  llm(prompt, ...) → dict
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None):
        key = "AIzaSyACuOqmThr8k7dIekDNJJH2OVWALT4ibb8"
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Set it as an environment variable or pass api_key= to GeminiLLM()."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    # ---- llama_cpp-compatible __call__ ----
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> dict:
        """
        Generate a response from Gemini, returning a dict that mirrors
        the llama_cpp response format:
            {"choices": [{"text": "<generated text>"}]}
        """
        import google.api_core.exceptions as google_exceptions

        # Strip Llama-specific [INST] formats which can confuse Gemini
        clean_prompt = prompt.replace("[INST]", "").replace("[/INST]", "").strip()

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            stop_sequences=stop or [],
        )

        try:
            response = self.model.generate_content(
                clean_prompt,
                generation_config=generation_config,
            )
            # Extract the generated text
            try:
                text = response.text
            except (ValueError, AttributeError):
                # If safety filters blocked the response, return empty
                text = ""
        except google_exceptions.GoogleAPIError as e:
            # Re-raise so chat.py can handle rate limits / quotas properly
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        return {"choices": [{"text": text}]}
