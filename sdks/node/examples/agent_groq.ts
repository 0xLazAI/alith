import { Agent } from "alith";

const agent = new Agent({
  model: "llama3-70b-8192",
  apiKey: "Your API Key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
});
(async () => {
  console.log(await agent.prompt("Calculate 10 - 3"));
})();
