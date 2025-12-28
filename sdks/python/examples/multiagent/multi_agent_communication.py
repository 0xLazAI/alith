"""
Example: Agent Communication with Shared Memory

This example demonstrates how agents communicate through shared memory,
sending messages and sharing context.
"""

import os
from alith import Agent
from alith.multi_agent import (
    AgentRole,
    MultiAgent,
    SharedMemory,
)

# Create shared memory
memory = SharedMemory()

# Create agents
planner = MultiAgent(
    agent_id="planner",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),  # Set OPENAI_API_KEY environment variable
    ),
    role=AgentRole.PLANNER,
    description="Creates plans and strategies",
)

executor = MultiAgent(
    agent_id="executor",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.EXECUTOR,
    description="Executes plans",
    dependencies=["planner"],  # Depends on planner
)

reviewer = MultiAgent(
    agent_id="reviewer",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.REVIEWER,
    description="Reviews work",
)

# Register agents
memory.register_agent(planner.agent_id)
memory.register_agent(executor.agent_id)
memory.register_agent(reviewer.agent_id)

# Set initial context
memory.set_context("project", "Web3 AI Agent Framework")
memory.set_context("deadline", "2025-01-31")

# Step 1: Planner creates a plan
print("Step 1: Planner creating plan...")
print("=" * 80)

plan_result = planner.execute(
    prompt="Create a detailed plan for building a Web3 AI agent framework. Include key components and implementation steps.",
    context=memory.context,
)

print(f"Planner output:\n{plan_result.output}\n")

# Store result
memory.store_result(plan_result)

# Planner sends message to executor
plan_message = planner.send_message(
    content=f"I've created a plan. Here are the key steps:\n{plan_result.output[:500]}",
    recipient=executor.agent_id,
    metadata={"type": "plan", "status": "ready"},
)
memory.send_message(plan_message)

# Step 2: Executor receives message and executes
print("\nStep 2: Executor receiving plan and executing...")
print("=" * 80)

# Get messages for executor
executor_messages = memory.get_messages(recipient=executor.agent_id)

executor_result = executor.execute(
    prompt="Based on the plan you received, start implementing the first component. Focus on the core agent system.",
    context=memory.context,
    messages=executor_messages,
)

print(f"Executor output:\n{executor_result.output}\n")

memory.store_result(executor_result)

# Executor sends message to reviewer
executor_message = executor.send_message(
    content=f"I've completed the first component. Here's what I implemented:\n{executor_result.output[:500]}",
    recipient=reviewer.agent_id,
    metadata={"type": "implementation", "component": "core_agent"},
)
memory.send_message(executor_message)

# Step 3: Reviewer reviews
print("\nStep 3: Reviewer reviewing work...")
print("=" * 80)

reviewer_messages = memory.get_messages(recipient=reviewer.agent_id)

reviewer_result = reviewer.execute(
    prompt="Review the implementation you received. Check for quality, best practices, and provide feedback.",
    context=memory.context,
    messages=reviewer_messages,
)

print(f"Reviewer output:\n{reviewer_result.output}\n")

memory.store_result(reviewer_result)

# Reviewer sends feedback back to executor
feedback_message = reviewer.send_message(
    content=f"Review feedback:\n{reviewer_result.output[:500]}",
    recipient=executor.agent_id,
    metadata={"type": "feedback", "action": "improve"},
)
memory.send_message(feedback_message)

# Display communication summary
print("\n" + "=" * 80)
print("Communication Summary:")
print("=" * 80)

print(f"\nTotal messages exchanged: {len(memory.messages)}")
print("\nMessage flow:")
for msg in memory.messages:
    recipient_str = msg.recipient if msg.recipient else "all"
    print(f"  {msg.sender} -> {recipient_str}: {msg.content[:100]}...")

print("\nAgent results:")
for agent_id in [planner.agent_id, executor.agent_id, reviewer.agent_id]:
    result = memory.get_result(agent_id)
    if result:
        print(f"  {agent_id}: {result.status.value} ({result.execution_time:.2f}s)")

print("\nShared context:")
for key, value in memory.context.items():
    print(f"  {key}: {value}")







