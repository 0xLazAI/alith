"""
This script demonstrates how to create a stateful chat session
with an Alith Agent.

Unlike .prompt(), a .chat() session automatically manages and
remembers the conversation history, allowing for follow-up questions.
"""

import os
from alith import Agent
from dotenv import load_dotenv

# --- Agent Setup ---

# Load an API key (e.g., Groq, which is fast for chat)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = "groq/llama3-8b-8192"

# Fallback to Anthropic if Groq key isn't set
if not api_key:
    print("GROQ_API_KEY not found. Falling back to Anthropic Haiku.")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = "anthropic/claude-3-haiku-20240307"

if not api_key:
    raise EnvironmentError("No GROQ_API_KEY or ANTHROPIC_API_KEY found.")

# 1. Initialize the agent
agent = Agent(
    model=model_name,
    api_key=api_key,
    system_prompt="You are a helpful assistant. Keep your answers concise."
)

# 2. Create a chat session
# This 'chat' object will store the conversation history.
chat = agent.chat()

# --- Main Execution ---

def main():
    print("Starting interactive chat. Type 'exit' to end.")
    print("Try asking a question, and then a follow-up question.")
    print("Example: 'Who was the first US president?' -> 'When was he born?'")
    print("-" * 30)

    try:
        while True:
            # 3. Get user input
            prompt_text = input("You: ")
            
            if prompt_text.lower() == 'exit':
                print("Ending chat. Goodbye!")
                break
            
            # 4. Send the prompt to the chat session
            # The 'chat' object sends the new prompt *along with*
            # all the previous messages in the history.
            response = chat.prompt(prompt_text)
            
            # 5. Print the agent's response
            print(f"Agent: {response.text}")

    except KeyboardInterrupt:
        print("\nEnding chat. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()