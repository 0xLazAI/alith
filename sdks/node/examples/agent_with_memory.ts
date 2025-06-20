import { Agent, WindowBufferMemory } from "alith";

const agent = new Agent({
  model: "gpt-4",
  memory: new WindowBufferMemory(),
});
console.log(await agent.prompt("Calculate 10 - 3"));
console.log(await agent.prompt("Calculate 10 - 3 again"));
