from typing import List


def chunk_text(
    text: str, max_chunk_token_size: int = 200, overlap_percent: float = 0.0
) -> List[str]:
    """Chunks a natural language text into smaller pieces.

    ## Parameters
    * `text` - The natural language text to chunk.
    * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
    * `overlap_percent` - The percentage of overlap between chunks. Default is None.
    """
    from ._alith import chunk_text as _chunk_text

    return _chunk_text(text, max_chunk_token_size, overlap_percent)
