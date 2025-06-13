from duckduckgo_search import DDGS

from alith import Agent


def search(query: str) -> str:
    """
    DuckDuckGoSearch is a tool designed to perform search queries on the DuckDuckGo search engine.
    It takes a search query string as input and returns relevant search results.
    This tool is ideal for scenarios where real-time information from the internet is required,
    such as finding the latest news, retrieving detailed information on a specific topic, or verifying facts.
    """
    results = DDGS.text(query, max_results=10)
    if results:
        response = f"Top 10 results for '{query}':\n\n"
        for i, result in enumerate(results, start=1):
            response += f"{i}. {result['title']}\n   {result['link']}\n"
        return response
    else:
        return f"No results found for '{query}'."


agent = Agent(
    name="Search Bot Agent",
    model="gpt-4",
    preamble="""You are a searcher. When I ask questions about Web3, you can search from the Internet and answer them. When you encounter other questions, you can directly answer them.""",
    tools=[search],
)

# Main loop to interact with the user
if __name__ == "__main__":
    print("Welcome to the DuckDuckGo Search Tool with OpenAI!")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = agent.prompt(user_input)
        print(f"Assistant: {response}")
