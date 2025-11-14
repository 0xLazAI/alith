import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - environment variables can be set directly
    pass

# Check for required multimodal dependencies before importing
try:
    import clip
    import torch
    from PIL import Image
    import faiss
except ImportError as e:
    print("Error: Multimodal dependencies are not installed.")
    print("\nTo install them, run:")
    print("  pip install -e \".[multimodal]\"  # for local development")
    print("  pip install \"alith[multimodal]\"  # from PyPI")
    print(f"\nMissing package: {e.name if hasattr(e, 'name') else 'unknown'}")
    sys.exit(1)

from pathlib import Path
from alith import ClipEmbeddings, ImageFAISSStore, MultimodalAgent

imgs_folder = Path(__file__).parent.parent.parent.parent / "imgs"
if not imgs_folder.exists():
    print(f"Warning: Image folder not found at {imgs_folder}")
    print("Please update the 'imgs_folder' path in this script to point to your image directory.")
    sys.exit(1)

embeddings = ClipEmbeddings()
store = ImageFAISSStore(embeddings=embeddings, dimension=512)

for img_file in sorted(imgs_folder.iterdir()):
    if img_file.is_file() and img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
        store.save(str(img_file))

found_image = store.search("logo", limit=1)[0]

# Load API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("The GROQ_API_KEY environment variable is not set.")

agent = MultimodalAgent(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    store=store,
)

prompt = "Describe this logo image in detail."
print(f"User: {prompt}")
response = agent.prompt(prompt, images=[found_image])
print(f"Agent: {response}")
