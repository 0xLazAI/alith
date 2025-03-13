from langchain_core.language_models import LLM as _LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from typing import Any, Dict, Iterator, List, Optional
from alith import Agent


class LLM(_LLM):
    agent: Agent

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.agent.prompt(prompt)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in self.agent.prompt(prompt):
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "Alith"}

    @property
    def _llm_type(self) -> str:
        return "Alith"
