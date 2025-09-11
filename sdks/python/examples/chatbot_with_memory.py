from alith import Agent
from alith import WindowBufferMemory

API_KEY = "<- Your API Key ->"

Luna = Agent(
    # model="llama-3.3-70b-versatile",
    # base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY,
    memory=WindowBufferMemory(),
)   

RUN_CHATBOT = True
converstation_count = 0

while RUN_CHATBOT:
    user_input = input("You: ").strip()
    if user_input.lower() in ["bye", "exit", "quit"]:
        RUN_CHATBOT = False
        print("Luna: Bye!")
        break
    else:
        if converstation_count == 0:
            system_prompt = "You are helpful coding AI Assistant"
        else:
            system_prompt = ""
        luna_thoughts = Luna.prompt("system prompt: " + system_prompt + " \n\n" + "user input: " + user_input)
        print("Luna: " + luna_thoughts)
        converstation_count += 1
