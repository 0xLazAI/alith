from ._alith import chunk_text as _chunk_text
from typing import List


def chunk_text(
    text: str, max_chunk_token_size: int = 200, overlap_percent=0.01
) -> List[str]:
    """Chunks a natural language text into smaller pieces.

    ## Parameters
    * `text` - The natural language text to chunk.
    * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
    * `overlap_percent` - The percentage of overlap between chunks. Default is None.
    """
    return _chunk_text(text, max_chunk_token_size, overlap_percent)
