from alith import Agent

agent = Agent(
    model="model name",  # your preferred model
    api_key="Your api key",  # use your api key
    base_url="your base url"
)

print("Alith Chatbot is ready!")
print("Ask your question:")
print("Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.prompt(user_input)
    print("Bot:", response)