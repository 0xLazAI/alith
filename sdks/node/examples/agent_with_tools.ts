import { Agent } from "../src";
import * as dotenv from "dotenv";
dotenv.config();

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
      parameters: {
        type: "object",
        properties: {
          x: { type: "number", description: "The number to subtract from" },
          y: { type: "number", description: "The number to subtract" },
        },
        required: ["x", "y"],
      },
      // 🔑 make handler synchronous unless you really need async
      handler: async (...args: unknown[]) => {
        console.log("📥 Raw args from Alith:", args);

        let x: number, y: number;

        try {
          // Case 1: JSON string
          if (typeof args[0] === "string") {
            const parsed = JSON.parse(args[0]);
            x = parsed.x;
            y = parsed.y;
          }
          // Case 2: Direct numbers
          else if (typeof args[0] === "number" && typeof args[1] === "number") {
            x = args[0] as number;
            y = args[1] as number;
          } else {
            throw new Error("Invalid args format");
          }

          console.log(`✅ Subtracting ${y} from ${x}`);
          return x - y;
        } catch (err) {
          console.error("❌ Error parsing args:", err);
          return NaN;
        }
      },
    },
  ],
});

agent
  .prompt("Calculate 10 - 3")
  .then((result) => {
    console.log("🎯 Final Result:", result);
  })
  .catch((err) => {
    console.error("❌ Error:", err);
  });
