"""
Local LLM Wrapper — Mistral 7B via llama-cpp-python


Usage:
    llm = LocalLlamaLLM()
    response = llm(prompt, max_tokens=256, temperature=0.0)
    # response == {"choices": [{"text": "..."}]}
"""

import os
import sys

# ── Resolve model path from settings ──────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from config.settings import LOCAL_MODEL_PATH, N_GPU_LAYERS, N_CTX  # type: ignore
except ImportError:
    LOCAL_MODEL_PATH = os.path.join(
        PROJECT_ROOT, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    N_GPU_LAYERS = 20
    N_CTX = 2048


# ── Mistral Instruct prompt format ────────────────────────────────────────────
def _mistral_prompt(user_text: str) -> str:
    """
    Wrap raw text in Mistral v0.2 instruct format.
    <s>[INST] {user_text} [/INST]
    """
    return f"<s>[INST] {user_text.strip()} [/INST]"


class LocalLlamaLLM:
    """
    Wraps llama-cpp-python's Llama behind the same callable interface
    that the old GeminiLLM provided:  llm(prompt, ...) → dict

    Return format:
        {"choices": [{"text": "<generated text>"}]}
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
        verbose: bool = False,
    ):
        from llama_cpp import Llama  # type: ignore[import]

        _model_path = model_path or LOCAL_MODEL_PATH
        _n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else N_GPU_LAYERS
        _n_ctx = n_ctx if n_ctx is not None else N_CTX

        if not os.path.exists(_model_path):
            raise FileNotFoundError(
                f"Model file not found: {_model_path}\n"
                "Please place the GGUF model at the configured LOCAL_MODEL_PATH."
            )

        print(f"📦 Loading local model: {os.path.basename(_model_path)}")
        print(f"   n_gpu_layers={_n_gpu_layers}  n_ctx={_n_ctx}")

        self._llm = Llama(
            model_path=_model_path,
            n_gpu_layers=_n_gpu_layers,
            n_ctx=_n_ctx,
            n_threads=os.cpu_count() or 4,
            verbose=verbose,
        )
        print("✅ Local model loaded successfully.")

    # ── llama_cpp-compatible __call__ ─────────────────────────────────────────
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> dict:
        """
        Generate a response from the local Mistral model.

        Returns a dict that mirrors the expected format:
            {"choices": [{"text": "<generated text>"}]}
        """
        # Wrap in Mistral instruct format (strip any leftover [INST] first)
        clean = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
        formatted = _mistral_prompt(clean)

        default_stop = ["[INST]", "</s>", "---", "[QUESTION]"]
        stop_seqs = stop if stop is not None else default_stop

        try:
            result = self._llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_seqs,
                echo=False,
            )
            text: str = result["choices"][0]["text"].strip()  # type: ignore[index]
        except Exception as e:
            raise RuntimeError(f"Local model generation error: {e}") from e

        return {"choices": [{"text": text}]}


