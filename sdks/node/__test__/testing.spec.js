import test from "ava";

import { Agent } from "../dist/index.js";

test("test", (t) => {
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
        handler: (x, y) => {
          return x - y;
        },
      },
    ],
  });
  t.is(agent.model(), "gpt-4o-mini");
});
