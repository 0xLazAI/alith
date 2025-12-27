"""
Example: Parallel Agent Execution

This example demonstrates executing multiple agents in parallel
to gather different perspectives on the same topic.
"""

import os
from alith import Agent
from alith.multi_agent import (
    AgentRole,
    MultiAgent,
    ParallelAgents,
    ParallelTask,
    SharedMemory,
)

# Create specialized agents
tech_expert = MultiAgent(
    agent_id="tech_expert",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),  # Set OPENAI_API_KEY environment variable
    ),
    role=AgentRole.ANALYZER,
    description="Technology and technical analysis specialist",
    capabilities=["technology", "technical analysis", "implementation"],
)

business_expert = MultiAgent(
    agent_id="business_expert",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.ANALYZER,
    description="Business and market analysis specialist",
    capabilities=["business", "market analysis", "strategy"],
)

security_expert = MultiAgent(
    agent_id="security_expert",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.ANALYZER,
    description="Security and privacy specialist",
    capabilities=["security", "privacy", "risk assessment"],
)

# Create parallel executor
parallel = ParallelAgents(
    memory=SharedMemory(),
    max_workers=3,
)

# Create tasks
query = "Evaluate the adoption of AI agents in Web3 applications"

tasks = [
    ParallelTask(
        agent=tech_expert,
        prompt=f"{query}\n\nFocus on technical aspects, implementation challenges, and technical feasibility.",
        name="technical_analysis",
    ),
    ParallelTask(
        agent=business_expert,
        prompt=f"{query}\n\nFocus on business opportunities, market potential, and economic impact.",
        name="business_analysis",
    ),
    ParallelTask(
        agent=security_expert,
        prompt=f"{query}\n\nFocus on security considerations, privacy implications, and risk factors.",
        name="security_analysis",
    ),
]

# Execute in parallel
print("Executing agents in parallel...")
print("=" * 80)

results = parallel.execute(tasks, wait_for_all=True)

# Display results
for agent_id, result in results.items():
    print(f"\n{agent_id.upper()}")
    print(f"Status: {result.status.value}")
    print(f"Execution time: {result.execution_time:.2f}s")
    if result.output:
        print(f"Output:\n{result.output[:300]}...")
    print("-" * 80)

# Aggregate insights
print("\n" + "=" * 80)
print("AGGREGATED INSIGHTS:")
print("=" * 80)

all_outputs = "\n\n".join([r.output for r in results.values() if r.output])
print(f"Combined analysis from {len(results)} experts:")
print(all_outputs[:500] + "..." if len(all_outputs) > 500 else all_outputs)








