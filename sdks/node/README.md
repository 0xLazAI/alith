# Alith Node SDK

## Installation

```shell
npm install alith
```

## Quick Start

```typescript
import { Agent } from "alith";

const agent = new Agent({
  name: "A dummy Agent",
  model: "gpt-4o-mini",
  preamble:
    "You are a comedian here to entertain the user using humour and jokes.",
});
console.log(agent.prompt("Entertain me!"));
```

## Developing

- Install node.js
- Install cargo (for Rust code)

Install dependencies

```shell
npm install
```

Building

```shell
npm run build
```

Testing

```shell
npm test
```

Format

```shell
npm run format
```
