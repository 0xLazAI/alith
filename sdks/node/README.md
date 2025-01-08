# Alith Node SDK

## Installation

```shell
npm install alith
```

## Quick Start

```typescript
import { Agent } from "alith";

const agent = new Agent("A dummy Agent", "gpt-4o-mini");
console.log(agent.promt("What is the problem?"));
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
