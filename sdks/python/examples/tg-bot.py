import os

from telegram import Update
from telegram.ext import Application, CallbackContext, MessageHandler, filters

from alith import Agent

# Initialize Alith Agent
agent = Agent(
    name="Telegram Bot Agent",
    model="gpt-4",
    preamble="""You are an advanced AI assistant built by [Alith](https://github.com/0xLazAI/alith).""",
)

# Initialize Telegram Bot
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
app = Application.builder().token(bot_token).build()


# Define message handler
async def handle_message(update: Update, context: CallbackContext) -> None:
    # Use the agent to generate a response
    response = agent.prompt(update.message.text)
    # Send the reply back to the Telegram chat
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


# Add handlers to the application
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

# Start the bot
if __name__ == "__main__":
    app.run_polling()
