import os
import re

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    CallbackContext,
)

from langchain_community.document_loaders.github import GithubFileLoader
from langchain_text_splitters import MarkdownTextSplitter
from pymilvus import MilvusClient, model
from alith import Agent

# --------------------------------------------
# Constants
# --------------------------------------------

GITHUB_ACCESS_KEY = os.getenv("GITHUB_ACCESS_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GITHUB_REPO = "0xLazAI/alith"
DOC_RELATIVE_PATH = "website/src/content"

# --------------------------------------------
# Init Embeddings Model and Document Database
# --------------------------------------------

client = MilvusClient("alith.db")
client.create_collection(
    collection_name="alith",
    dimension=768,
)
# If connection to https://huggingface.co/ failed, uncomment the following path.
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Note: This will download a small embedding model "paraphrase-albert-small-v2" (~50MB) from Hugging Face.
embedding_fn = model.DefaultEmbeddingFunction()


def create_vector_store():
    docs = GithubFileLoader(
        repo=GITHUB_REPO,
        access_token=GITHUB_ACCESS_KEY,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: re.match(
            f"{DOC_RELATIVE_PATH}/.*\\.mdx?", file_path
        )
        is not None,
    ).load()
    text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = [split.page_content for split in text_splitter.split_documents(docs)]
    vectors = embedding_fn.encode_documents(docs)
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]
    client.insert(collection_name="alith", data=data)


create_vector_store()

# --------------------------------------------
# Init Alith Agent
# --------------------------------------------

agent = Agent(
    name="Telegram Bot Agent",
    model="gpt-4",
    preamble="""You are a comedian here to entertain the user using humour and jokes.""",
)


def prompt_with_rag(text: str) -> str:
    query_vectors = embedding_fn.encode_queries([text])
    # Search from the vector database
    res = client.search(
        collection_name="alith",
        data=query_vectors,
        limit=2,
        output_fields=["text", "subject"],
    )
    docs = [d["entity"]["text"] for r in res for d in r]
    response = agent.prompt(
        "{}\n\n<attachments>\n{}</attachments>\n".format(text, "".join(docs))
    )
    return response


# --------------------------------------------
# Init Telegram Bot
# --------------------------------------------


async def handle_message(update: Update, context: CallbackContext) -> None:
    response = prompt_with_rag(update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

# Start the bot
if __name__ == "__main__":
    app.run_polling()
