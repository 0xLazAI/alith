"""
This script shows an example of using the Alith Agent
to create a Math Solver assistant using OpenAI or Anthropic models.

Usage:
- Make sure you have installed the `alith` library.
- Set your API key (e.g., Anthropic or OpenAI) as an environment variable.
- Run this example:
    python examples/agent_math_solver.py
"""

import os
from alith import Agent

# Load your API key from environment variable
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

# Initialize the agent
agent = Agent(
    model="gpt-4o-mini",  # or "claude-3-opus-20240229"
    api_key=api_key,
    base_url="https://api.openai.com/v1",  # change if using another provider
    preamble="You are a helpful AI tutor who explains and solves math problems step-by-step."
)

def main():
    """Main function that runs the math solver agent."""
    print("\n🔹 Example: Math Problem Solver\n")
    problem = "Solve for x: 2x + 5 = 13"
    print(f"Question: {problem}\n")

    try:
        response = agent.prompt(problem)
        print("Answer:", response)
    except Exception as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    main()
