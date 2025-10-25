"""
This script provides an example of how to use the Alith library
to create an AI agent powered by Anthropic's Claude 3 models.

Usage:
- Make sure you have the 'alith' library installed.
- Set your Anthropic API key as an environment variable:
  export ANTHROPIC_API_KEY='your_api_key_here'
- Run the script: python -m examples.agent_anthropic
"""

import os
from alith import Agent

# Best practice is to load your API key from environment variables.
# You can get a key from https://console.anthropic.com/
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("The ANTHROPIC_API_KEY environment variable is not set.")

# 1. Initialize the Agent with an Anthropic model configuration.
# We are using Claude 3 Haiku here because it's the fastest model.
# Other popular options: "claude-3-sonnet-20240229" or "claude-3-opus-20240229"
agent = Agent(
    model="claude-3-haiku-20240307",
    api_key=api_key,
    base_url="https://api.anthropic.com/v1",
    preamble="You are a helpful assistant who is an expert in science. Explain complex topics in a simple, easy-to-understand way.",
)

# 2. Define a main function to run the agent.
def main():
    """
    Runs the agent with a sample prompt and prints the output.
    """
    prompt_text = "Why is the sky blue?"
    
    print(f"-> Asking the agent: '{prompt_text}'")
    
    try:
        response = agent.prompt(prompt_text)
        print("\n✨ Agent's Response:")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

# 3. Execute the main function when the script is run.
if __name__ == "__main__":
    main()