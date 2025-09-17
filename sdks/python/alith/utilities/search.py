# ============================================================================
# SEARCH UTILITIES
# ============================================================================

from typing import Any, Dict, List, Literal, Callable
from pydantic import BaseModel, Field
from ..tool import Tool


class DuckDuckGoTool:
    """
    Unified DuckDuckGo Tool.

    - source:
        * "text"   → general web search
        * "news"   → news search
        * "images" → image search
        * "videos" → video search
    - output:
        * structured=True  → structured list in 'results'
        * structured=False → stitched text in 'answer'
    """

    class Params(BaseModel):
        query: str = Field(..., description="Search query string.")
        source: Literal["text", "news", "images", "videos"] = Field(
            "text", description="Where to search: text, news, images, or videos."
        )
        structured: bool = Field(
            True,
            description="If True, return structured {'results': [...]}; if False, return stitched text {'answer': '...'}."
        )
        max_results: int = Field(
            5, ge=1, le=50, description="Maximum number of results to return."
        )

    # -------------------------
    # Helpers
    # -------------------------
    _source_funcs: Dict[str, str] = {
        "text": "text",
        "news": "news",
        "images": "images",
        "videos": "videos",
    }

    _mappers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        "text": lambda r: {
            "title": r.get("title"),
            "link": r.get("href"),
            "snippet": r.get("body"),
        },
        "news": lambda r: {
            "title": r.get("title"),
            "link": r.get("url"),
            "source": r.get("source"),
            "date": r.get("date"),
        },
        "images": lambda r: {
            "title": r.get("title"),
            "image": r.get("image"),
            "thumbnail": r.get("thumbnail"),
            "source": r.get("source"),
            "link": r.get("url"),
        },
        "videos": lambda r: {
            "title": r.get("title"),
            "link": r.get("content"),
            "source": r.get("source"),
            "description": r.get("description"),
        },
    }

    @staticmethod
    def _format_answer_text(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "No results found."
        parts = [
            f"{it.get('title','')}\n{it.get('snippet') or it.get('description','')}\nSource: {it.get('link') or it.get('image','')}"
            for it in items
        ]
        return "\n\n---\n\n".join(parts)

    # -------------------------
    # Core handler
    # -------------------------
    @staticmethod
    def _handler(
        query: str,
        source: str = "text",
        max_results: int = 5,
        structured: bool = True,
    ) -> Dict[str, Any]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            return {
                "structured": structured,
                "results": None,
                "answer": None,
                "error": "Please install with: pip install ddgs (or duckduckgo-search)",
            }

        func_name = DuckDuckGoTool._source_funcs.get(source)
        mapper = DuckDuckGoTool._mappers.get(source, lambda x: x)

        with DDGS() as ddg:
            raw = list(getattr(ddg, func_name)(query, max_results=max_results)) if func_name else []

        items = [mapper(r) for r in raw]

        if structured:
            return {"structured": True, "results": items, "answer": None}
        else:
            return {"structured": False, "results": None, "answer": DuckDuckGoTool._format_answer_text(items)}

    # -------------------------
    # Direct execution helpers
    # -------------------------
    def run(
        self,
        query: str,
        *,
        source: str = "text",
        max_results: int = 5,
        structured: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the DuckDuckGo search.

        Args:
            query: The search query.
            source: The source to search.
            max_results: The maximum number of results to return.
            structured: Whether to return structured results.
        """
        return DuckDuckGoTool._handler(
            query=query,
            source=source,
            max_results=max_results,
            structured=structured,
        )

    __call__ = run

    def text(self, query: str, *, max_results: int = 5, structured: bool = True) -> Dict[str, Any]:
        """
        Run the DuckDuckGo search.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            structured: Whether to return structured results.
        """
        return self.run(query, source="text", max_results=max_results, structured=structured)

    def news(self, query: str, *, max_results: int = 5, structured: bool = True) -> Dict[str, Any]:
        """
        Run the DuckDuckGo search.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            structured: Whether to return structured results.
        """
        return self.run(query, source="news", max_results=max_results, structured=structured)

    def images(self, query: str, *, max_results: int = 5, structured: bool = True) -> Dict[str, Any]:
        """
        Run the DuckDuckGo search.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            structured: Whether to return structured results.
        """
        return self.run(query, source="images", max_results=max_results, structured=structured)

    def videos(self, query: str, *, max_results: int = 5, structured: bool = True) -> Dict[str, Any]:
        """
        Run the DuckDuckGo search.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            structured: Whether to return structured results.
        """
        return self.run(query, source="videos", max_results=max_results, structured=structured)

    # -------------------------
    # Integration with Tool system
    # -------------------------
    def to_tool(self) -> Tool:
        """
        Return the DuckDuckGo tool.

        Returns:
            The DuckDuckGo tool.
        """
        return Tool(
            name="duckduckgo",
            description="DuckDuckGo search (text, news, images, videos). Outputs structured results or stitched text.",
            parameters=DuckDuckGoTool.Params,
            handler=DuckDuckGoTool._handler,
        )
