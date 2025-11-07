"""
This script provides an example of how to use the Alith library
to create an AI agent powered by Cohere's Command R model.
"""

import os
from alith import Agent
from dotenv import load_dotenv

# Best practice is to load your API key from environment variables.
# You can get a key from https://dashboard.cohere.com/
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise EnvironmentError("COHERE_API_KEY environment variable is not set.")

# 1. Initialize the Agent with a Cohere model configuration.
# We are using 'command-r' which is a strong, general-purpose model.
agent = Agent(
    model="cohere/command-r",
    api_key=api_key,
    system_prompt="You are a helpful assistant who explains complex topics simply."
)

# 2. Define a main function to run the agent.
def main():
    # Run the agent with a simple prompt and print the output.
    prompt_text = "What is the difference between a B-Tree and a B+ Tree?"

    print(f"User: {prompt_text}")

    try:
        # 3. The .prompt() method sends the text and waits for a response.
        response = agent.prompt(prompt_text)
        
        # 4. Print the agent's text response and token usage.
        print(f"Agent: {response.text}")
        print("\n--- Usage ---")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 5. Standard Python entry point
if __name__ == "__main__":
    main()