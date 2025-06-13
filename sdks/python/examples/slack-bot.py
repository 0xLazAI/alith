import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from alith import Agent

# Initialize Slack Bolt app with the bot token
slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))
agent = Agent(
    name="Slack Bot Agent",
    model="gpt-4",
    preamble="""You are an advanced AI assistant built by [Alith](https://github.com/0xLazAI/alith).""",
)


# Define a message handler
@slack_app.message("")
def handle_message(message, say):
    # Use the agent to generate a response
    response = agent.prompt(message["text"])
    # Send the reply back to the Slack channel
    say(response)


# Start the bot using Socket Mode
if __name__ == "__main__":
    handler = SocketModeHandler(slack_app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
