import { Agent } from "alith";

const agent = new Agent({
  model: "deepseek-chat", // or deepseek-reasoner for DeepSeek R1
  apiKey: "<Your API Key>",
  baseUrl: "api.deepseek.com",
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
});
(async () => {
  console.log(await agent.prompt("Calculate 10 - 3"));
})();
