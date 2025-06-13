import os
import re

from langchain_community.document_loaders.github import GithubFileLoader
from telegram import Update
from telegram.ext import Application, CallbackContext, MessageHandler, filters

from alith import Agent, MilvusStore, chunk_text

# --------------------------------------------
# Constants
# --------------------------------------------

GITHUB_ACCESS_KEY = os.getenv("GITHUB_ACCESS_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GITHUB_REPO = "0xLazAI/alith"
DOC_RELATIVE_PATH = "website/src/content"

# --------------------------------------------
# Init Document Database
# --------------------------------------------


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
    docs = chunk_text(docs, overlap_percent=0.2)
    return MilvusStore().save_docs(docs)


# --------------------------------------------
# Init Alith Agent
# --------------------------------------------

agent = Agent(
    name="Telegram Bot Agent",
    model="gpt-4",
    preamble="""You are a comedian here to entertain the user using humour and jokes.""",
    store=create_vector_store(),
)

# --------------------------------------------
# Init Telegram Bot
# --------------------------------------------


async def handle_message(update: Update, context: CallbackContext) -> None:
    response = agent.prompt(update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

# Start the bot
if __name__ == "__main__":
    app.run_polling()
