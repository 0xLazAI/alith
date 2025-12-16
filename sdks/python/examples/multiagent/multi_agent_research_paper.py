"""
Example: Research Paper Writing Workflow

A complete workflow for writing a research paper using multiple agents:
1. Research (gather information)
2. Analyze (identify key points)
3. Outline (create structure)
4. Write sections (parallel writing)
5. Review and refine
"""

import os
from alith import Agent
from alith.multi_agent import (
    AgentChain,
    AgentRole,
    ChainStep,
    MultiAgent,
    ParallelAgents,
    ParallelTask,
    SharedMemory,
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

analyzer = MultiAgent(
    agent_id="analyzer",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.ANALYZER,
)

planner = MultiAgent(
    agent_id="planner",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.PLANNER,
)

writer_intro = MultiAgent(
    agent_id="writer_intro",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.WRITER,
)

writer_body = MultiAgent(
    agent_id="writer_body",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.WRITER,
)

writer_conclusion = MultiAgent(
    agent_id="writer_conclusion",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.WRITER,
)

reviewer = MultiAgent(
    agent_id="reviewer",
    agent=Agent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    role=AgentRole.REVIEWER,
)

# Create shared memory
memory = SharedMemory()

# Step 1: Research and Analysis Chain
print("Phase 1: Research and Analysis")
print("=" * 80)

research_chain = AgentChain(
    steps=[
        ChainStep(
            agent=researcher,
            prompt_template="Research the following topic thoroughly: {input}\n\nGather comprehensive information from multiple perspectives.",
        ),
        ChainStep(
            agent=analyzer,
            prompt_template="Analyze this research: {previous_output}\n\nIdentify key themes, important findings, and main arguments.",
        ),
        ChainStep(
            agent=planner,
            prompt_template="Based on this analysis: {previous_output}\n\nCreate a detailed outline for a research paper with sections: Introduction, Body (with subsections), and Conclusion.",
        ),
    ],
    memory=memory,
)

topic = "The role of decentralized AI agents in Web3 ecosystems"
research_results = research_chain.execute(topic)

print(f"Research completed. Outline created.")
outline = research_chain.get_final_result()
if outline:
    print(f"\nOutline preview:\n{outline.output[:300]}...")

# Step 2: Parallel Writing
print("\n\nPhase 2: Parallel Section Writing")
print("=" * 80)

parallel = ParallelAgents(memory=memory, max_workers=3)

writing_tasks = [
    ParallelTask(
        agent=writer_intro,
        prompt=f"Write an introduction section for a research paper on: {topic}\n\nUse this outline as guidance: {outline.output if outline else 'N/A'}\n\nMake it engaging and set up the research context.",
    ),
    ParallelTask(
        agent=writer_body,
        prompt=f"Write the main body section for a research paper on: {topic}\n\nUse this outline as guidance: {outline.output if outline else 'N/A'}\n\nInclude detailed analysis, examples, and supporting evidence.",
    ),
    ParallelTask(
        agent=writer_conclusion,
        prompt=f"Write a conclusion section for a research paper on: {topic}\n\nUse this outline as guidance: {outline.output if outline else 'N/A'}\n\nSummarize key findings and provide future directions.",
    ),
]

writing_results = parallel.execute(writing_tasks)

print("\nWriting completed for all sections:")
for agent_id, result in writing_results.items():
    print(f"  {agent_id}: {result.status.value} ({result.execution_time:.2f}s)")

# Step 3: Combine and Review
print("\n\nPhase 3: Review and Refinement")
print("=" * 80)

# Combine all sections
all_sections = "\n\n".join([
    result.output for result in writing_results.values() if result.output
])

review_result = reviewer.execute(
    prompt=f"Review this complete research paper:\n\n{all_sections}\n\nProvide feedback on:\n1. Overall coherence\n2. Quality of arguments\n3. Areas for improvement\n4. Suggestions for refinement.",
    context=memory.context,
)

memory.store_result(review_result)

print("Review completed:")
print(f"\nReview feedback:\n{review_result.output[:400]}...")

# Final output
print("\n" + "=" * 80)
print("FINAL RESEARCH PAPER")
print("=" * 80)
print(all_sections)

print("\n" + "=" * 80)
print("Workflow Summary")
print("=" * 80)
print(f"Total agents used: 7")
print(f"Total steps: 3 phases")
print(f"Total execution time: {sum(r.execution_time for r in research_results) + sum(r.execution_time for r in writing_results.values()) + review_result.execution_time:.2f}s")
print(f"Memory messages: {len(memory.messages)}")
print(f"Memory results: {len(memory.results)}")






