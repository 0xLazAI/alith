"""
Example: Advanced Agent Orchestrator

This example demonstrates a complex workflow with:
- Sequential steps
- Parallel execution
- Conditional branching
- Result aggregation
"""

import os
from alith import Agent
from alith.multi_agent import (
    AgentOrchestrator,
    AgentRole,
    MultiAgent,
    SharedMemory,
    WorkflowCondition,
    WorkflowStep,
)

# Create agents
researcher = MultiAgent(
    agent_id="researcher",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),  # Set OPENAI_API_KEY environment variable
    ),
    role=AgentRole.RESEARCHER,
)

coder = MultiAgent(
    agent_id="coder",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.CODER,
)

reviewer = MultiAgent(
    agent_id="reviewer",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.REVIEWER,
)

validator = MultiAgent(
    agent_id="validator",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.VALIDATOR,
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    memory=SharedMemory(),
    max_parallel_workers=2,
)

# Define workflow steps

# Step 1: Research
orchestrator.add_step(
    WorkflowStep(
        name="research",
        agents=researcher,
        prompt="Research the following topic and provide comprehensive information: {input}",
        next_step="plan",
    )
)

# Step 2: Plan (could be parallel with multiple planners)
orchestrator.add_step(
    WorkflowStep(
        name="plan",
        agents=researcher,  # Reuse researcher for planning
        prompt="Based on this research: {previous_output}\n\nCreate a detailed implementation plan with steps.",
        next_step="code",
    )
)

# Step 3: Code
orchestrator.add_step(
    WorkflowStep(
        name="code",
        agents=coder,
        prompt="Based on this plan: {previous_output}\n\nWrite clean, well-documented code to implement the plan.",
        next_step="review",
    )
)

# Step 4: Review (parallel review from multiple perspectives)
orchestrator.add_step(
    WorkflowStep(
        name="review",
        agents=[reviewer, validator],
        prompt={
            reviewer.agent_id: "Review this code: {previous_output}\n\nCheck for code quality, best practices, and potential improvements.",
            validator.agent_id: "Validate this code: {previous_output}\n\nCheck for errors, security issues, and correctness.",
        },
        parallel=True,
        next_step="finalize",
    )
)

# Step 5: Finalize
orchestrator.add_step(
    WorkflowStep(
        name="finalize",
        agents=coder,
        prompt="Based on these reviews: {previous_output}\n\nIncorporate feedback and create the final version.",
    )
)

# Execute workflow
print("Executing complex workflow...")
print("=" * 80)

initial_input = "Create a Python function to calculate Fibonacci numbers with memoization"

result = orchestrator.execute(
    initial_input=initial_input,
    context={"language": "Python", "focus": "performance"},
)

# Display results
print("\nWorkflow Execution Summary:")
print("=" * 80)
print(f"Success: {result['success']}")
print(f"Steps executed: {len(result['execution_history'])}")
print(f"\nFinal Output:")
print(result['final_output'])

print("\n" + "=" * 80)
print("Execution History:")
print("=" * 80)

for i, step in enumerate(result['execution_history'], 1):
    print(f"\nStep {i}: {step['step']}")
    for agent_id, agent_result in step['results'].items():
        print(f"  {agent_id}: {agent_result['status']}")
        if agent_result.get('output'):
            print(f"    Preview: {agent_result['output'][:150]}...")

print("\n" + "=" * 80)
print("Memory Summary:")
print("=" * 80)
print(f"Registered agents: {result['memory_summary']['registered_agents']}")
print(f"Total messages: {result['memory_summary']['total_messages']}")
print(f"Total results: {result['memory_summary']['total_results']}")








