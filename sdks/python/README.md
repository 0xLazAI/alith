# Alith Python SDK

## Installation

```shell
python3 -m pip install alith
```

## Quick Start

- Simple Agent

```python
from alith import Agent

agent = Agent(
    name="A dummy Agent",
    model="gpt-4o-mini",
    preamble="You are a comedian here to entertain the user using humour and jokes.",
)
print(agent.prompt("Entertain me!"))
```

- Agent with Tools

```python
from alith import Agent


def sum(x: int, y: int) -> int:
    """Add x and y together"""
    x + y


def sub(x: int, y: int) -> int:
    """Subtract y from x (i.e.: x - y)"""
    x + y


agent = Agent(
    name="Calculator Agent",
    model="gpt-4o-mini",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    tools=[sum, sub],
)
print(agent.prompt("Calculate 10 - 3"))
```

## Examples

See [here](./examples/README.md) for more examples.

## Developing

Setup virtualenv:

```shell
python3 -m venv venv
```

Activate venv:

```shell
source venv/bin/activate
```

Install maturin:

```shell
cargo install maturin
```

Build bindings:

```shell
maturin develop
```

Test

```shell
python3 -m pytest
```
