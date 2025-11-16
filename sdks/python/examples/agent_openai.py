"""
This example demonstrates how to create a simple agent using the Alith library
with an OpenAI model. The agent answers user questions in a clear and concise way.

Usage:
1. Install dependencies:
       pip install alith openai

2. Export your OpenAI API key:
       export OPENAI_API_KEY="your_key_here"

3. Run the example:
       python examples/agent_openai.py
"""

import os
from alith import Agent

def load_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return key


def main():
    api_key = load_api_key()

    # Create the agent with an OpenAI model configuration
    agent = Agent(
        model="gpt-4.1-mini",
        api_key=api_key,
        base_url="https://api.openai.com/v1/",
        preamble="You are a helpful assistant skilled at explaining concepts simply.",
    )

    prompt = "How does a rainbow form?"

    print(f"\nUser: {prompt}\n")

    try:
        response = agent.prompt(prompt)
        print("Agent:", response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
