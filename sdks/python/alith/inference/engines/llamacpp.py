try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from importlib.util import find_spec
from io import StringIO
from pathlib import Path

HUGGINGFACE_HUB_AVAILABLE = find_spec("huggingface_hub") is not None


class LlamaEngine:
    llm: Llama
    model: str

    def __init__(self, model_path: str, **kwargs):
        """Create a Llama inference engine from a local or remote model name or path."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama_cpp is not installed. Please install it with: "
                "python3 -m pip install llama_cpp_python or find more installation information at https://pypi.org/project/llama-cpp-python/"
            )
        if Path(model_path).exists():
            self.llm = Llama(model_path=model_path, **kwargs)
            self.model = model_path
        else:
            if not HUGGINGFACE_HUB_AVAILABLE:
                raise ImportError(
                    "huggingface-hub is not installed. Please install it with: "
                    "python3 -m pip install huggingface-hub"
                )
            self.llm = Llama.from_pretrained(model_path, **kwargs)
            self.model = model_path

    def prompt(self, prompt: str) -> str:
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        if is_completion_response(resp):
            return resp["choices"][0]["message"]["content"]
        result = StringIO()
        for chunk in resp:
            result.write(chunk["choices"][0]["message"]["content"])
        result.seek(0)
        result.getvalue()


def is_completion_response(resp):
    return (
        isinstance(resp, dict)
        and "choices" in resp
        and isinstance(resp["choices"], list)
        and len(resp["choices"]) > 0
        and "message" in resp["choices"][0]
    )
