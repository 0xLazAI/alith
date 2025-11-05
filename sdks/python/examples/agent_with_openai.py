"""
This script provides an example of how to use the Alith library
to create an AI agent powered by OpenAI’s GPT model.
"""

import os
from alith import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

# 1. Initialize the Agent with OpenAI model
agent = Agent(
    model="openai/gpt-4o-mini",  # or gpt-4, gpt-3.5-turbo, etc.
    api_key=api_key,
    system_prompt="You are an AI assistant that explains topics clearly and concisely."
)

# 2. Define the main function
def main():
    prompt_text = "Explain the difference between synchronous and asynchronous programming in Python."

    print(f"User: {prompt_text}\n")

    try:
        # 3. Use the agent to get a response
        response = agent.prompt(prompt_text)

        # 4. Print the result
        print("Agent:", response.text)
        print("\n--- Token Usage ---")
        print(f"Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"Completion Tokens: {response.usage.completion_tokens}")
        print(f"Total Tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 5. Entry point
if __name__ == "__main__":
    main()
