import { Agent } from "alith";

const agent = new Agent({
  name: "A twitter agent",
  model: "gpt-4",
  preamble: "You are a automatic twitter agent.",
  mcpConfigPath: "mcp_twitter.json",
});
console.log(await agent.prompt("Search Twitter for tweets about AI"));
console.log(
  await agent.prompt('Post a tweet saying "Hello from Alith Twitter Agent!"')
);
console.log(await agent.prompt("Get the latest tweets from @OpenAI"));
console.log(await agent.prompt("Chat with Grok about quantum computing"));
