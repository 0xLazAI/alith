# Alith Python SDK

## Installation

```shell
python3 -m pip install alith
```

## Quick Start

```python
from alith import Agent

agent = Agent(
    name="A dummy Agent",
    model="gpt-4o-mini",
)

agent.promt("What is the problem?")
```

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
