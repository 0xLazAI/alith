/**
 * Alith LazAI Inference Server
 * 
 * Start the node with the PRIVATE_KEY environment variable set.
 * For OpenAI/ChatGPT API, set OPENAI_API_KEY.
 * For other OpenAI-compatible APIs, set LLM_API_KEY and LLM_BASE_URL.
 * 
 * Example curl request:
 * curl http://localhost:8000/v1/chat/completions \
 *   -H "Content-Type: application/json" \
 *   -H "X-LazAI-User: 0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194" \
 *   -H "X-LazAI-Nonce: 123456" \
 *   -H "X-LazAI-Signature: HSDGYUSDOWP123" \
 *   -H "X-LazAI-Token-ID: 1" \
 *   -d '{
 *     "model": "gpt-3.5-turbo",
 *     "messages": [
 *       {"role": "system", "content": "You are a helpful assistant"},
 *       {"role": "user", "content": "What is the capital of France?"}
 *     ],
 *     "temperature": 0.7,
 *     "max_tokens": 100
 *   }'
 */

import express, { Request, Response, Express } from "express";
import cors from "cors";
import axios from "axios";
import { Client } from "../lazai/client";
import { INFERENCE_TYPE, validateRequest } from "../lazai/request";

const client = new Client();
const app: Express = express();

app.use(cors());
app.use(express.json());

// Initialize OpenAI-compatible client
const getOpenAIConfig = () => {
  const apiKey = process.env.OPENAI_API_KEY || process.env.LLM_API_KEY || "";
  const baseURL = process.env.OPENAI_BASE_URL || process.env.LLM_BASE_URL || "https://api.openai.com/v1";
  
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY or LLM_API_KEY environment variable is required");
  }
  
  return { apiKey, baseURL };
};

/**
 * POST /v1/chat/completions
 */
app.post("/v1/chat/completions", async (req: Request, res: Response) => {
  try {
    const { apiKey, baseURL } = getOpenAIConfig();
    const response = await axios.post(
      `${baseURL}/chat/completions`,
      req.body,
      {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      }
    );
    return res.json(response.data);
  } catch (error) {
    console.error("Chat completion error:", error);
    if (axios.isAxiosError(error)) {
      return res.status(error.response?.status || 500).json({
        error: {
          message: error.response?.data?.error?.message || error.message,
          type: error.response?.data?.error?.type || "internal_error",
        },
      });
    }
    return res.status(500).json({
      error: {
        message: error instanceof Error ? error.message : "Unknown error",
        type: "internal_error",
      },
    });
  }
});

/**
 * POST /v1/embeddings
 */
app.post("/v1/embeddings", async (req: Request, res: Response) => {
  try {
    const { apiKey, baseURL } = getOpenAIConfig();
    const response = await axios.post(
      `${baseURL}/embeddings`,
      req.body,
      {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      }
    );
    return res.json(response.data);
  } catch (error) {
    console.error("Embedding error:", error);
    if (axios.isAxiosError(error)) {
      return res.status(error.response?.status || 500).json({
        error: {
          message: error.response?.data?.error?.message || error.message,
          type: error.response?.data?.error?.type || "internal_error",
        },
      });
    }
    return res.status(500).json({
      error: {
        message: error instanceof Error ? error.message : "Unknown error",
        type: "internal_error",
      },
    });
  }
});

/**
 * GET /v1/models
 */
app.get("/v1/models", async (req: Request, res: Response) => {
  try {
    const { apiKey, baseURL } = getOpenAIConfig();
    const response = await axios.get(
      `${baseURL}/models`,
      {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
        },
      }
    );
    return res.json(response.data);
  } catch (error) {
    console.error("Get models error:", error);
    if (axios.isAxiosError(error)) {
      return res.status(error.response?.status || 500).json({
        error: {
          message: error.response?.data?.error?.message || error.message,
          type: error.response?.data?.error?.type || "internal_error",
        },
      });
    }
    return res.status(500).json({
      error: {
        message: error instanceof Error ? error.message : "Unknown error",
        type: "internal_error",
      },
    });
  }
});

/**
 * Run the inference server
 */
export function run(
  host: string = "localhost",
  port: number = 8000,
  settlement: boolean = false
): void {
  if (settlement) {
    app.use(async (req: Request, res: Response, next: () => void) => {
      try {
        await validateRequest(req.headers as Record<string, string>, INFERENCE_TYPE, client);
        next();
      } catch (error) {
        return res.status(401).json({
          error: {
            message: error instanceof Error ? error.message : "Unauthorized",
            type: "unauthorized_error",
          },
        });
      }
    });
  }

  app.listen(port, host, () => {
    console.log(`Alith LazAI Inference Server running on http://${host}:${port}`);
  });
}

