# Alith Multi-Agent Framework

A comprehensive framework for building multi-agent systems with Alith.

## Overview

The Multi-Agent Framework enables you to:
- Create sequential agent chains where each agent builds on previous output
- Execute multiple agents in parallel for concurrent processing
- Enable agent communication through shared memory
- Orchestrate complex workflows with conditional branching
- Manage agent roles and specializations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              AgentOrchestrator                          │
│         (Complex Workflow Management)                    │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│  AgentChain    │            │  ParallelAgents     │
│  (Sequential)  │            │  (Concurrent)       │
└───────┬────────┘            └─────────┬──────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │    SharedMemory       │
            │  (Communication Hub)  │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │     MultiAgent         │
            │   (Role-based Agent)   │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │    Alith Agent         │
            │    (Core LLM)          │
            └────────────────────────┘
```

## Core Components

### 1. MultiAgent
Wrapper around Alith Agent with role-based behavior and multi-agent features.

**Features:**
- Agent identification and tracking
- Role-based preambles
- Capability matching
- Dependency management
- Message creation

**Example:**
```python
from alith import Agent
from alith.multi_agent import MultiAgent, AgentRole

agent = MultiAgent(
    agent_id="researcher",
    agent=Agent(model="gpt-4o-mini"),
    role=AgentRole.RESEARCHER,
    description="Research specialist",
    capabilities=["research", "analysis"],
)
```

### 2. AgentChain
Sequential chain of agents where each agent processes the output of the previous agent.

**Features:**
- Sequential execution
- Output passing between steps
- Conditional step execution
- Output transformation
- Error handling

**Example:**
```python
from alith.multi_agent import AgentChain, ChainStep

chain = AgentChain(
    steps=[
        ChainStep(
            agent=researcher,
            prompt_template="Research: {input}",
        ),
        ChainStep(
            agent=writer,
            prompt_template="Write about: {previous_output}",
        ),
    ],
)

results = chain.execute("AI in Web3")
```

### 3. ParallelAgents
Execute multiple agents concurrently for independent tasks.

**Features:**
- Concurrent execution
- Thread pool management
- Result aggregation
- Callback support
- Timeout handling

**Example:**
```python
from alith.multi_agent import ParallelAgents, ParallelTask

parallel = ParallelAgents(max_workers=3)

tasks = [
    ParallelTask(agent=tech_expert, prompt="Analyze technology..."),
    ParallelTask(agent=business_expert, prompt="Analyze business..."),
    ParallelTask(agent=security_expert, prompt="Analyze security..."),
]

results = parallel.execute(tasks)
```

### 4. SharedMemory
Centralized memory for agent communication and context sharing.

**Features:**
- Message broadcasting and routing
- Result storage and retrieval
- Context management
- Agent registration
- History management

**Example:**
```python
from alith.multi_agent import SharedMemory

memory = SharedMemory()

# Send message
message = agent1.send_message("Hello", recipient="agent2")
memory.send_message(message)

# Get messages
messages = memory.get_messages(recipient="agent2")

# Store context
memory.set_context("project", "Web3 AI")
```

### 5. AgentOrchestrator
Advanced orchestrator for complex multi-agent workflows.

**Features:**
- Sequential and parallel steps
- Conditional branching
- Dynamic workflow construction
- Result transformation
- Execution history

**Example:**
```python
from alith.multi_agent import AgentOrchestrator, WorkflowStep

orchestrator = AgentOrchestrator()

orchestrator.add_step(
    WorkflowStep(
        name="research",
        agents=researcher,
        prompt="Research: {input}",
        next_step="analyze",
    )
)

result = orchestrator.execute("Topic to research")
```

## Agent Roles

Predefined roles with default preambles:

- **RESEARCHER**: Research specialist
- **WRITER**: Content writer
- **ANALYZER**: Data analyst
- **CODER**: Software engineer
- **REVIEWER**: Quality reviewer
- **TRANSLATOR**: Language translator
- **SUMMARIZER**: Content summarizer
- **PLANNER**: Strategic planner
- **EXECUTOR**: Task executor
- **VALIDATOR**: Result validator
- **CUSTOM**: Custom role (no default preamble)

## Usage Patterns

### Pattern 1: Simple Chain
```python
chain = AgentChain([
    ChainStep(agent=researcher, prompt_template="Research: {input}"),
    ChainStep(agent=writer, prompt_template="Write: {previous_output}"),
])
results = chain.execute("Topic")
```

### Pattern 2: Parallel Execution
```python
parallel = ParallelAgents()
tasks = [
    ParallelTask(agent=agent1, prompt="Task 1"),
    ParallelTask(agent=agent2, prompt="Task 2"),
]
results = parallel.execute(tasks)
```

### Pattern 3: Communication
```python
memory = SharedMemory()
message = agent1.send_message("Info", recipient="agent2")
memory.send_message(message)
messages = memory.get_messages(recipient="agent2")
```

### Pattern 4: Complex Workflow
```python
orchestrator = AgentOrchestrator()
orchestrator.add_step(WorkflowStep(...))
orchestrator.add_step(WorkflowStep(...))
result = orchestrator.execute(input)
```

## Best Practices

1. **Use Appropriate Roles**: Assign agents roles that match their tasks
2. **Manage Dependencies**: Define agent dependencies for proper execution order
3. **Share Context**: Use SharedMemory for context that multiple agents need
4. **Handle Errors**: Check AgentResult.status for failures
5. **Optimize Parallelization**: Use parallel execution for independent tasks
6. **Use Chains for Dependencies**: Use chains when agents depend on previous output

## Examples

See the `examples/multiagent/` directory for complete examples:
- `multi_agent_chain.py`: Sequential chain example
- `multi_agent_parallel.py`: Parallel execution example
- `multi_agent_orchestrator.py`: Complex workflow example
- `multi_agent_communication.py`: Agent communication example
- `multi_agent_research_paper.py`: Complete workflow example

## Integration with Alith

The Multi-Agent Framework integrates seamlessly with other Alith features:

- **Stores**: Use vector stores for RAG in agents
- **Tools**: Agents can use tools
- **Memory**: Integrate with Alith memory systems
- **Embeddings**: Use embeddings for semantic search
- **LazAI**: Integrate with blockchain features

## Performance Considerations

- **Parallel Execution**: Use for independent tasks to reduce total time
- **Chain Execution**: Sequential chains take longer but enable dependencies
- **Memory Management**: Limit message/result history to avoid memory issues
- **Thread Pool**: Adjust max_workers based on your system capabilities

## Future Enhancements

Planned features:
- Async/await support for better concurrency
- Agent persistence and state management
- Advanced workflow patterns (loops, conditionals)
- Agent monitoring and observability
- Distributed agent execution

## License

Part of the Alith project. See main LICENSE file.






