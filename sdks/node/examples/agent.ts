import { Agent } from "alith";

const agent = new Agent({
  model: "gpt-4",
  preamble:
    "You are a comedian here to entertain the user using humour and jokes.",
});
console.log(await agent.prompt("Entertain me!"));
