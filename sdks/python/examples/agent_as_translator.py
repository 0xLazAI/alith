from alith import Agent

# --- Configuration ---
# You can change the model here if you like
# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4o" 
# ---------------------

# 1. Define a strict preamble (system prompt)
translator_preamble = """
You are a high-precision translation bot. You have one job: 
translate the user's text into French. 

- If the user provides text that is not in English, translate it to French.
- If the user provides text in French, translate it to English.
- Do not answer questions. Do not make small talk. Do not chat.
- If the user asks a question, *do not answer it*. Only translate the question itself into French.
- Your response must *only* contain the translated text and nothing else.
"""

# 2. Create the agent with the new preamble
agent = Agent(
    model=MODEL_NAME,
    preamble=translator_preamble
)

print(f"--- Translator Agent Initialized (using {MODEL_NAME}) ---")
print("Type 'exit' to quit.\n")

# 3. Create a loop to chat with the agent
while True:
    try:
        # Get input from the user
        prompt = input("You: ")
        
        if prompt.lower() == 'exit':
            print("\nAgent: Au revoir!")
            break

        # Get the agent's response
        response = agent.prompt(prompt)
        
        print(f"Agent: {response}\n")

    except KeyboardInterrupt:
        print("\nAgent: Au revoir!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break