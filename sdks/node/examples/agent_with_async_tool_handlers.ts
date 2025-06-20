import { Agent } from "alith";

const agent = new Agent({
  model: "deepseek-ai/DeepSeek-V3",
  baseUrl: "api.siliconflow.cn/v1",
  apiKey: process.env.LLM_API_KEY,
  tools: [
    {
      name: "subtract",
      description: "Subtract y from x (i.e.: x - y)",
      parameters: {
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
      },
      handler: async (x: number, y: number) => {
        return x - y;
      },
    },
  ],
});
console.log(await agent.prompt("Calculate 10 - 3"));
