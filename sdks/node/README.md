# Alith Node SDK

## Installation

- Install `alith` dependency

```shell
npm install alith
# Or use pnpm `pnpm install alith`
# Or use yarn `yarn install alith`
```

- Install the `json-schema` dependency

```shell
npm i --save-dev @types/json-schema
# Or use pnpm `pnpm install --save-dev @types/json-schema`
# Or use yarn `yarn install --save-dev @types/json-schema`
```

## Quick Start

- Simple Agent

```typescript
import { Agent } from "alith";

const agent = new Agent({
  model: "gpt-4o-mini",
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
});
console.log(await agent.prompt("Calculate 10 - 3"));
```

- Agent with Tools

```typescript
import { Agent } from "alith";

const agent = new Agent({
  model: "gpt-4o-mini",
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
  tools: [
    {
      name: "subtract",
      description: "Subtract y from x (i.e.: x - y)",
      parameters: JSON.stringify({
        type: "object",
        properties: {
          x: {
            type: "number",
            description: "The number to substract from",
          },
          y: {
            type: "number",
            description: "The number to substract",
          },
        },
      }),
      handler: (x: number, y: number) => {
        return x - y;
      },
    },
  ],
});
console.log(await agent.prompt("Calculate 10 - 3"));
```

## Examples

See [here](./examples/README.md) for more examples.

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
