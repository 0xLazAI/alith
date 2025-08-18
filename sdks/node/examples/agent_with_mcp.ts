import { Agent } from "alith";

const agent = new Agent({
  model: "gpt-4",
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
  mcpConfigPath: "servers_config.json",
});
console.log(await agent.prompt("Calculate 10 - 3"));
