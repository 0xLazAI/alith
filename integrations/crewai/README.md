# Alith Integration for CrewAI

Enable high-performance, Web3-friendly multi-agent workflows by powering CrewAI with Alith agents.

## 🚀 Features

- **High-Performance Inference**: Leverage Alith's Rust-based core for 2-3x faster inference
- **Web3-Native**: Built-in support for wallets, blockchain, and smart contract interactions
- **Privacy-First**: TEE integration for sensitive data in multi-agent workflows
- **Cost-Effective**: Optimized for smaller, faster models
- **Drop-in Replacement**: Use Alith as an LLM backend with minimal code changes

## 📦 Installation

```bash
# Install from PyPI (when published)
pip install alith-crewai

# Or install from source
cd integrations/crewai/python
pip install -e .

# With optional CrewAI tools
pip install alith-crewai[tools]
```

## 🎯 Quick Start

```python
from crewai import Crew, Task
from alith_crewai import create_alith_crew_agent

# Create Alith-powered agent
researcher = create_alith_crew_agent(
    role="Research Analyst",
    goal="Find and analyze information",
    backstory="Expert researcher with 10 years experience",
    model="llama-3.3-70b-versatile"
)

# Create task
task = Task(
    description="Research AI agent frameworks",
    agent=researcher,
    expected_output="Summary of AI agent frameworks"
)

# Create and run crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

## 🔧 Configuration

### Environment Variables

```bash
# Required: API key for your LLM provider
export GROQ_API_KEY="your-groq-api-key"

# Or use OpenAI
export OPENAI_API_KEY="your-openai-key"
```

### Custom Models

```python
# Use different models
agent = create_alith_crew_agent(
    role="Analyst",
    goal="Analyze data",
    backstory="Data expert",
    model="llama-3.1-8b-instant",  # Faster, cheaper model
    api_key="your-key",
    base_url="https://api.groq.com/openai/v1"
)
```

## 📚 Examples

### Simple Two-Agent Crew

```python
from crewai import Crew, Task
from alith_crewai import create_alith_crew_agent

# Create agents
researcher = create_alith_crew_agent(
    role="Researcher",
    goal="Find information",
    backstory="Expert researcher"
)

writer = create_alith_crew_agent(
    role="Writer",
    goal="Create content",
    backstory="Talented writer"
)

# Create tasks
research = Task(description="Research AI trends", agent=researcher)
write = Task(description="Write article about AI trends", agent=writer)

# Run crew
crew = Crew(agents=[researcher, writer], tasks=[research, write])
result = crew.kickoff()
```

### Web3 Research Crew

See [`examples/web3_research_crew.py`](./examples/web3_research_crew.py) for a complete example of using Alith's Web3 capabilities in a multi-agent workflow.

### Using Existing Alith Agent

```python
from alith import Agent
from alith_crewai import create_alith_agent_with_custom_llm

# Create pre-configured Alith agent
alith_agent = Agent(
    model="llama-3.3-70b-versatile",
    preamble="You are an expert analyst",
    api_key="your-key"
)

# Use it in CrewAI
crew_agent = create_alith_agent_with_custom_llm(
    alith_agent=alith_agent,
    role="Analyst",
    goal="Analyze data",
    backstory="Expert analyst"
)
```

## 🔄 Tool Conversion

Convert tools between CrewAI and Alith formats:

```python
from alith import Tool as AlithTool
from crewai_tools import SerperDevTool
from alith_crewai import convert_crewai_tool_to_alith, convert_alith_tool_to_crewai

# CrewAI → Alith
crewai_tool = SerperDevTool()
alith_tool = convert_crewai_tool_to_alith(crewai_tool)

# Alith → CrewAI
alith_tool = AlithTool(
    name="search",
    description="Search the web",
    handler=lambda query: f"Results for: {query}"
)
crewai_tool = convert_alith_tool_to_crewai(alith_tool)
```

## 🎨 Why Use Alith with CrewAI?

| Feature | Native CrewAI | With Alith |
|---------|--------------|------------|
| **Performance** | Standard | 2-3x faster |
| **Web3 Support** | Manual setup | Built-in |
| **Privacy** | Basic | TEE integration |
| **Cost** | Standard | Optimized |
| **Inference** | Python | Rust core |

## 📖 API Reference

### `create_alith_crew_agent`

Create a CrewAI agent powered by Alith.

**Parameters:**
- `role` (str): Agent's role
- `goal` (str): Agent's goal
- `backstory` (str): Agent's background
- `model` (str, optional): LLM model name (default: "llama-3.3-70b-versatile")
- `api_key` (str, optional): API key (defaults to GROQ_API_KEY env var)
- `base_url` (str, optional): Base URL for API (defaults to Groq API)
- `tools` (list, optional): List of tools (Alith or CrewAI format)
- `verbose` (bool, optional): Enable verbose logging (default: False)
- `**kwargs`: Additional CrewAI agent parameters

**Returns:** CrewAI Agent instance

### `AlithLLM`

Custom LLM class wrapping Alith agents.

**Methods:**
- `call(messages, **kwargs)`: Main inference method

### Tool Converters

- `convert_crewai_tool_to_alith(tool)`: Convert CrewAI tool to Alith
- `convert_alith_tool_to_crewai(tool)`: Convert Alith tool to CrewAI

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint
ruff check .

# Type check
mypy alith_crewai
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTOR.md](../../../CONTRIBUTOR.md) for guidelines.

## 📄 License

Apache 2.0 - see [LICENSE](../../../LICENSE)

## 🔗 Links

- [Alith Documentation](https://alith.lazai.network/docs)
- [CrewAI Documentation](https://docs.crewai.com)
- [GitHub Repository](https://github.com/0xLazAI/alith)
- [LazAI Network](https://lazai.network)

## 💬 Support

- [Telegram](https://t.me/alithai)
- [X/Twitter](https://x.com/0xalith)
- [GitHub Issues](https://github.com/0xLazAI/alith/issues)
