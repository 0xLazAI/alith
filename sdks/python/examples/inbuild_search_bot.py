import os
from alith import Agent, DuckDuckGoTool


def create_search_agent(api_key: str, base_url: str) -> Agent:
    ddg_tool = DuckDuckGoTool().to_tool()
    return Agent(
        name="Search Assistant",
        model="llama-3.3-70b-versatile",
        base_url=base_url,
        api_key=api_key,
        preamble="You are a helpful search assistant. Use search tools to find relevant information.",
        tools=[ddg_tool]
    )


def create_summary_agent(api_key: str, base_url: str) -> Agent:
    return Agent(
        name="Content Summarizer",
        model="llama-3.3-70b-versatile",
        base_url=base_url,
        api_key=api_key,
    )


def main():
    API_KEY = os.getenv("GROQ_API_KEY")
    BASE_URL = "https://api.groq.com/openai/v1"
    
    search_agent = create_search_agent(API_KEY, BASE_URL)
    summary_agent = create_summary_agent(API_KEY, BASE_URL)
    
    query = "learn python programming"
    search_prompt = f"Find 3 relevant YouTube links for: {query}. Provide links with descriptions."
    
    search_results = search_agent.prompt(search_prompt)
    summary_prompt = f"Make this more concise and professional: {search_results}"
    summary = summary_agent.prompt(summary_prompt)
    
    print("SEARCH RESULTS:")
    print(search_results)
    print("\nSUMMARY:")
    print(summary)


if _name_ == "_main_":
    main()