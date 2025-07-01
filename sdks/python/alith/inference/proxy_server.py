import os
import sys
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Alith Inference Proxy Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
oai = openai.OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


@app.post("/v1/chat/completions")
async def completions(request: Request):
    try:
        request_data = await request.json()
        return oai.chat.completions.create(**request_data)
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        request_data = await request.json()
        return oai.embeddings.create(**request_data)
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def models():
    try:
        return oai.models.list()
    except Exception as e:
        logger.error(f"Get remote model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
