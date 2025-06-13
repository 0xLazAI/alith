import tempfile

import requests


def download_file(url: str) -> str:
    """Download the file and return the temporary file path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".enc")
        with open(temp_file.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return temp_file.name
    except Exception:
        raise
