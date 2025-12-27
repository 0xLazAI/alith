# Multi-Agent Framework Examples

This directory contains examples demonstrating the Alith Multi-Agent Framework capabilities.

## Overview

The Multi-Agent Framework provides:
- **Sequential Chains**: Agents execute in sequence, each building on previous output
- **Parallel Execution**: Multiple agents work simultaneously
- **Shared Memory**: Agents communicate through messages and shared context
- **Orchestration**: Complex workflows with conditional branching

## Examples

### 1. `multi_agent_chain.py`
**Sequential Agent Chain**

Demonstrates a simple research -> analyze -> write workflow where each agent processes the output of the previous agent.

**Usage:**
```bash
python examples/multi_agent_chain.py
```

**Key Features:**
- Sequential execution
- Output passing between agents
- Shared memory for context

### 2. `multi_agent_parallel.py`
**Parallel Agent Execution**

Shows how to execute multiple agents in parallel to gather different perspectives on the same topic.

**Usage:**
```bash
python examples/multi_agent_parallel.py
```

**Key Features:**
- Concurrent agent execution
- Independent task processing
- Result aggregation

### 3. `multi_agent_orchestrator.py`
**Advanced Orchestrator**

Demonstrates a complex workflow with sequential steps, parallel execution, and conditional logic.

**Usage:**
```bash
python examples/multi_agent_orchestrator.py
```

**Key Features:**
- Complex workflow orchestration
- Parallel and sequential steps
- Conditional branching
- Result transformation

### 4. `multi_agent_communication.py`
**Agent Communication**

Shows how agents communicate through shared memory, sending messages and sharing context.

**Usage:**
```bash
python examples/multi_agent_communication.py
```

**Key Features:**
- Message passing between agents
- Shared context management
- Agent dependencies
- Communication patterns

### 5. `multi_agent_research_paper.py`
**Complete Workflow Example**

A real-world example: writing a research paper using multiple agents in a multi-phase workflow.

**Usage:**
```bash
python examples/multi_agent_research_paper.py
```

**Key Features:**
- Multi-phase workflow
- Chain + parallel execution
- Real-world use case
- Complete workflow demonstration

## Quick Start

1. **Install dependencies:**
```bash
pip install alith
```

2. **Set your API key:**
Edit the example files and replace `"your-api-key"` with your actual API key.

3. **Run an example:**
```bash
python examples/multi_agent_chain.py
```

## Framework Components

### Core Classes

- **`MultiAgent`**: Wrapper around Alith Agent with role-based behavior
- **`AgentChain`**: Sequential chain of agents
- **`ParallelAgents`**: Parallel agent executor
- **`SharedMemory`**: Communication and context sharing
- **`AgentOrchestrator`**: Advanced workflow orchestration

### Agent Roles

Predefined roles available:
- `RESEARCHER`: Research specialist
- `WRITER`: Content writer
- `ANALYZER`: Data analyst
- `CODER`: Software engineer
- `REVIEWER`: Quality reviewer
- `TRANSLATOR`: Language translator
- `SUMMARIZER`: Content summarizer
- `PLANNER`: Strategic planner
- `EXECUTOR`: Task executor
- `VALIDATOR`: Result validator

## Architecture

```
┌─────────────────────────────────────────┐
│         AgentOrchestrator               │
│  (Complex workflow management)          │
└─────────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
┌───▼────┐        ┌─────▼─────┐
│ Chain  │        │  Parallel  │
│(Seq)   │        │ (Concurrent)│
└───┬────┘        └─────┬─────┘
    │                   │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   SharedMemory    │
    │ (Communication)   │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │    MultiAgent      │
    │  (Role-based)      │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Alith Agent      │
    │  (Core LLM)        │
    └────────────────────┘
```

## Best Practices

1. **Use appropriate roles**: Assign agents roles that match their tasks
2. **Manage dependencies**: Define agent dependencies for proper execution order
3. **Share context**: Use SharedMemory for context that multiple agents need
4. **Handle errors**: Check AgentResult.status for failures
5. **Optimize parallelization**: Use parallel execution for independent tasks
6. **Use chains for dependencies**: Use chains when agents depend on previous output

## Advanced Usage

### Custom Agent Roles

```python
from alith.multi_agent import MultiAgent, AgentRole

custom_agent = MultiAgent(
    agent_id="custom",
    agent=Agent(model="gpt-4o-mini"),
    role=AgentRole.CUSTOM,
    description="Custom role agent",
    capabilities=["custom_task"],
)
```

### Conditional Execution

```python
from alith.multi_agent import WorkflowCondition

def check_quality(results):
    return all(r.status == AgentStatus.COMPLETED for r in results.values())

condition = WorkflowCondition(
    check=check_quality,
    description="All agents completed successfully",
)
```

### Result Transformation

```python
def aggregate_results(results):
    return "\n\n".join([r.output for r in results.values() if r.output])

step = WorkflowStep(
    name="aggregate",
    agents=[agent1, agent2],
    prompt="...",
    transform=aggregate_results,
)
```

## Troubleshooting

**Issue**: Agents not receiving messages
- **Solution**: Ensure agents are registered in SharedMemory before sending messages

**Issue**: Parallel execution not working
- **Solution**: Check that tasks are independent and don't share mutable state

**Issue**: Chain stops early
- **Solution**: Check `stop_on_error` parameter and handle errors in conditions

## Next Steps

- Explore the framework source code in `alith/multi_agent/`
- Create your own workflows
- Integrate with other Alith features (stores, tools, etc.)
- Build production multi-agent systems

## Support

For issues and questions:
- Check the main Alith documentation
- Review example code
- Open an issue on GitHub






