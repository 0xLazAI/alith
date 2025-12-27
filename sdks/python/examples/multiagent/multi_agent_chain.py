"""
Example: Sequential Agent Chain

This example demonstrates a research -> analyze -> write workflow
where each agent builds on the previous agent's output.
"""

import os
from alith import Agent
from alith.multi_agent import (
    AgentChain,
    AgentRole,
    ChainStep,
    MultiAgent,
    SharedMemory,
)

# Create agents with specific roles
researcher = MultiAgent(
    agent_id="researcher",
    agent=Agent(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),  # Set GROQ_API_KEY environment variable
    ),
    role=AgentRole.RESEARCHER,
    description="Researches topics and gathers information",
)

analyzer = MultiAgent(
    agent_id="analyzer",
    agent=Agent(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    role=AgentRole.ANALYZER,
    description="Analyzes research and identifies key insights",
)

writer = MultiAgent(
    agent_id="writer",
    agent=Agent(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    role=AgentRole.WRITER,
    description="Writes content based on analysis",
)

# Create chain steps
chain = AgentChain(
    steps=[
        ChainStep(
            agent=researcher,
            prompt_template="Research the following topic: {input}\n\nProvide comprehensive information about this topic.",
            name="research",
        ),
        ChainStep(
            agent=analyzer,
            prompt_template="Analyze the following research: {previous_output}\n\nIdentify the key insights, trends, and important points.",
            name="analyze",
        ),
        ChainStep(
            agent=writer,
            prompt_template="Based on this analysis: {previous_output}\n\nWrite a clear, engaging article that presents these insights in an accessible way.",
            name="write",
        ),
    ],
    memory=SharedMemory(),
)

# Execute the chain
print("Executing agent chain...")
print("=" * 80)

topic = "The impact of AI on Web3 development"
results = chain.execute(topic)

# Display results
for i, result in enumerate(results, 1):
    print(f"\nStep {i}: {result.agent_id}")
    print(f"Status: {result.status.value}")
    print(f"Execution time: {result.execution_time:.2f}s")
    if result.output:
        print(f"Output preview: {result.output[:200]}...")
    print("-" * 80)

# Get final result
final_result = chain.get_final_result()
if final_result:
    print("\n" + "=" * 80)
    print("FINAL OUTPUT:")
    print("=" * 80)
    print(final_result.output)






