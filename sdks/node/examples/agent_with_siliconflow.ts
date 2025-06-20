import { Agent } from "alith";

const agent = new Agent({
  model: "deepseek-ai/DeepSeek-V3",
  baseUrl: "api.siliconflow.cn/v1",
  apiKey: process.env.LLM_API_KEY,
});
console.log(await agent.prompt("Calculate 10 - 3"));
