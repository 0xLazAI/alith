import { Agent } from "alith";

async function main() {
  const agent = new Agent({
    model: "gpt-4",
    preamble:
      "You are a comedian here to entertain the user using humour and jokes.",
  });
  console.log(await agent.prompt("Entertain me!"));
}

main().catch(console.error);
