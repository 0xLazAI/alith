"""
This script provides an example of how to use the Alith Client
to interact with a local Large Language Model (LLM) running via Ollama.

Ollama (https://ollama.com/) allows you to run open-source LLMs locally.

Before running this script:
1. Install Ollama: Go to https://ollama.com/download
2. Pull a model (e.g., Llama 3): ollama pull llama3
3. Ensure Ollama is running in the background.

Usage:
- Make sure you have the 'alith' library installed.
- Run the script: python -m examples.agent_with_ollama
"""

import os
import requests # Used to check if Ollama is running
from alith import Agent # For using Agent functionality, though we'll focus on Client concept

# We're simulating a 'Client' interaction for demonstration.
# In a real-world 'Client' scenario (like in agent_with_lazai.py),
# you might interact with a specific service or blockchain.
# For this example, we'll configure an Agent with Ollama's API.

# --- Configuration for Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Ensure a model like 'llama3' is pulled in Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

def check_ollama_status():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        print(f"Ollama server is running at {OLLAMA_BASE_URL}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"Error: Ollama server not reachable at {OLLAMA_BASE_URL}. Is it running?")
        print("Please start Ollama and ensure a model like 'llama3' is pulled (`ollama pull llama3`).")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Error checking Ollama status: {e}. Response: {response.text}")
        return False
    except requests.exceptions.Timeout:
        print(f"Error: Connection to Ollama at {OLLAMA_BASE_URL} timed out.")
        return False


def main():
    """
    Demonstrates creating an Alith Agent to interact with a local Ollama LLM.
    """
    if not check_ollama_status():
        print("Exiting as Ollama is not ready.")
        return

    print(f"\n--- Alith Agent with Local Ollama ({OLLAMA_MODEL}) Example ---")

    # Initialize the Agent to use the Ollama API.
    # Note: For Ollama, the 'apiKey' is often not required or can be a dummy value.
    # The 'baseUrl' is crucial for pointing to the local Ollama server.
    # Initialize the Agent with the updated, correct parameter names
    agent = Agent(
        model=OLLAMA_MODEL,
        api_key="ollama-key-not-needed",
        base_url=OLLAMA_BASE_URL,
        preamble=f"You are a local AI assistant powered by {OLLAMA_MODEL}. Provide concise and helpful answers.",
    )

    prompt_text = "What is the capital of France?"

    print(f"-> Asking the local Ollama agent: '{prompt_text}'")

    try:
        response = agent.prompt(prompt_text)
        print("\n✨ Agent's Response (from Ollama):")
        print(response)

        print("\n--- Another example ---")
        prompt_text_2 = "Explain the concept of 'lazy evaluation' in programming."
        print(f"-> Asking the local Ollama agent: '{prompt_text_2}'")
        response_2 = agent.prompt(prompt_text_2)
        print("\n✨ Agent's Response (from Ollama):")
        print(response_2)

    except Exception as e:
        print(f"\nAn error occurred during agent interaction: {e}")
        print("Ensure the specified model is pulled in Ollama (e.g., `ollama pull llama3`)")


if __name__ == "__main__":
    main()