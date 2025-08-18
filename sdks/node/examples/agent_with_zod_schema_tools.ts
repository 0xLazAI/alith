import { Agent } from "../src";
import { z } from "zod";
import * as dotenv from "dotenv";
dotenv.config();

export const InputSchema = z
  .object({
    x: z.number().describe("The number to subtract from"),
    y: z.number().describe("The number to subtract"),
  })
  .strip();

const agent = new Agent({
  model: "deepseek-ai/DeepSeek-V3",
  baseUrl: "api.siliconflow.cn/v1",
  apiKey: process.env.LLM_API_KEY,
  preamble:
    "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
  tools: [
    {
      name: "subtract",
      description: "Subtract y from x (i.e.: x - y)",
      parameters: InputSchema,
      handler: async (...args: unknown[]) => {
        console.log("📥 Raw args from Alith:", args);

        let x: number, y: number;
        if (typeof args[0] === "string") {
          const parsed = JSON.parse(args[0]);
          x = parsed.x;
          y = parsed.y;
        } else if (typeof args[0] === "number" && typeof args[1] === "number") {
          x = args[0] as number;
          y = args[1] as number;
        } else {
          throw new Error("Invalid args format");
        }

        const result = x - y;
        console.log(`✅ Subtracting ${y} from ${x} = ${result}`);
        return String(result); // ✅ return a string (serializable, no Promise)
      },
    },
  ],
});

// Always await inside async function
(async () => {
  try {
    const result = await agent.prompt("Calculate 10 - 3");
    console.log("🎯 Final Result:", result);
    process.exit(0);
  } catch (err) {
    console.error("❌ Error:", err);
    process.exit(1);
  }
})();
