/**
 * Alith LazAI Inference Server with AEON BNB x402 Payment Integration
 * 
 * This example demonstrates how to integrate AEON BNB x402 SDK with LazAI API server
 * to enable native x402 payments on BNB Chain.
 * 
 * Prerequisites:
 * 1. Install AEON BNB x402 SDK: npm install @aeon-project/bnb-x402
 * 2. Set PRIVATE_KEY environment variable
 * 3. Set OPENAI_API_KEY or LLM_API_KEY environment variable
 * 4. Set X402_FACILITATOR_URL (defaults to https://facilitator.aeon.xyz)
 * 
 * Example curl request with x402 payment:
 * curl http://localhost:8000/v1/chat/completions \
 *   -H "Content-Type: application/json" \
 *   -H "X-402-Payment-Proof: <payment_proof>" \
 *   -d '{
 *     "model": "gpt-3.5-turbo",
 *     "messages": [{"role": "user", "content": "Hello"}]
 *   }'
 */

import dotenv from "dotenv";
dotenv.config();

import express, { Request, Response, Express } from "express";
import cors from "cors";
import axios from "axios";
import { Client } from "../src/lazai/client";
import { INFERENCE_TYPE, validateRequest } from "../src/lazai/request";

/**
 * AEON BNB x402 SDK Integration
 * 
 * To use the actual AEON BNB x402 SDK:
 * 1. Install the SDK: npm install @aeon-project/bnb-x402
 *    (or follow installation instructions from: https://github.com/AEON-Project/bnb-x402)
 * 
 * 2. Replace the X402Client placeholder class below with the actual SDK import:
 *    import { X402Client } from "@aeon-project/bnb-x402";
 * 
 * 3. Update the X402Client usage to match the actual SDK API
 * 
 * SDK Documentation: https://github.com/AEON-Project/bnb-x402/blob/master/README.md
 * Facilitator Endpoint: https://facilitator.aeon.xyz
 */
interface X402Payment {
  amount: string;
  currency: string;
  recipient: string;
  proof: string;
}

interface X402VerificationResult {
  valid: boolean;
  payment?: X402Payment;
  error?: string;
}

// Placeholder for x402 SDK - replace with actual import when SDK is available
// import { X402Client, verifyPayment } from "@aeon-project/bnb-x402";
class X402Client {
  private facilitatorUrl: string;

  constructor(facilitatorUrl: string = "https://facilitator.aeon.xyz") {
    this.facilitatorUrl = facilitatorUrl;
  }

  /**
   * Verify x402 payment proof
   * This is a placeholder - replace with actual SDK implementation
   */
  async verifyPayment(proof: string): Promise<X402VerificationResult> {
    try {
      // In actual implementation, this would call the AEON Facilitator API
      const response = await axios.post(
        `${this.facilitatorUrl}/verify`,
        { proof },
        {
          headers: { "Content-Type": "application/json" },
        }
      );

      if (response.data.valid) {
        return {
          valid: true,
          payment: response.data.payment,
        };
      }

      return {
        valid: false,
        error: response.data.error || "Invalid payment proof",
      };
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : "Verification failed",
      };
    }
  }

  /**
   * Create payment challenge (HTTP 402 response)
   */
  createPaymentChallenge(amount: string, currency: string = "USDT"): {
    status: number;
    headers: Record<string, string>;
    body: any;
  } {
    return {
      status: 402,
      headers: {
        "X-402-Payment-Required": "true",
        "X-402-Amount": amount,
        "X-402-Currency": currency,
        "X-402-Facilitator": this.facilitatorUrl,
      },
      body: {
        error: {
          type: "payment_required",
          message: `Payment required: ${amount} ${currency}`,
          amount,
          currency,
          facilitator: this.facilitatorUrl,
        },
      },
    };
  }
}

// Validate required environment variables
const PRIVATE_KEY = process.env.PRIVATE_KEY;
if (!PRIVATE_KEY) {
  console.error("❌ Error: PRIVATE_KEY environment variable is required");
  process.exit(1);
}

const client = new Client();
const app: Express = express();
const x402Client = new X402Client(process.env.X402_FACILITATOR_URL);

app.use(cors());
app.use(express.json());

// Configuration
const PAYMENT_AMOUNT = process.env.X402_PAYMENT_AMOUNT || "0.01";
const PAYMENT_CURRENCY = process.env.X402_PAYMENT_CURRENCY || "USDT";
const X402_ENABLED = process.env.X402_ENABLED !== "false"; // Default to enabled

// Initialize OpenAI-compatible client
let _loggedOpenAIConfig = false;
const getOpenAIConfig = () => {
  const apiKey = process.env.OPENAI_API_KEY || process.env.LLM_API_KEY || "";
  const baseURL = process.env.OPENAI_BASE_URL || process.env.LLM_BASE_URL || "https://api.openai.com/v1";

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY or LLM_API_KEY environment variable is required");
  }

  if (!_loggedOpenAIConfig) {
    console.log(`🔗 LLM Base URL: ${baseURL}`);
    console.log(`🔐 LLM API Key present: ${apiKey ? "yes" : "no"}`);
    _loggedOpenAIConfig = true;
  }

  return { apiKey, baseURL };
};

/**
 * Middleware to verify x402 payment
 */
async function verifyX402Payment(req: Request, res: Response, next: () => void) {
  if (!X402_ENABLED) {
    return next();
  }

  const paymentProof = req.headers["x-402-payment-proof"] as string;

  if (!paymentProof) {
    const challenge = x402Client.createPaymentChallenge(PAYMENT_AMOUNT, PAYMENT_CURRENCY);
    return res.status(challenge.status).set(challenge.headers).json(challenge.body);
  }

  try {
    const verification = await x402Client.verifyPayment(paymentProof);

    if (!verification.valid) {
      const challenge = x402Client.createPaymentChallenge(PAYMENT_AMOUNT, PAYMENT_CURRENCY);
      return res.status(challenge.status).set(challenge.headers).json({
        ...challenge.body,
        error: {
          ...challenge.body.error,
          message: verification.error || "Invalid payment proof",
        },
      });
    }

    // Payment verified - attach payment info to request for logging/accounting
    (req as any).x402Payment = verification.payment;
    next();
  } catch (error) {
    console.error("x402 payment verification error:", error);
    const challenge = x402Client.createPaymentChallenge(PAYMENT_AMOUNT, PAYMENT_CURRENCY);
    return res.status(challenge.status).set(challenge.headers).json({
      ...challenge.body,
      error: {
        ...challenge.body.error,
        message: "Payment verification failed",
      },
    });
  }
}

/**
 * POST /v1/chat/completions with x402 payment
 */
app.post("/v1/chat/completions", verifyX402Payment, async (req: Request, res: Response) => {
  try {
    const { apiKey, baseURL } = getOpenAIConfig();
    
    // Log payment if x402 is enabled
    if (X402_ENABLED && (req as any).x402Payment) {
      console.log(`💰 x402 Payment received: ${(req as any).x402Payment.amount} ${(req as any).x402Payment.currency}`);
    }

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
 * POST /v1/embeddings with x402 payment
 */
app.post("/v1/embeddings", verifyX402Payment, async (req: Request, res: Response) => {
  try {
    const { apiKey, baseURL } = getOpenAIConfig();
    
    if (X402_ENABLED && (req as any).x402Payment) {
      console.log(`💰 x402 Payment received: ${(req as any).x402Payment.amount} ${(req as any).x402Payment.currency}`);
    }

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
app.get("/v1/models", async (_req: Request, res: Response) => {
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
 * GET /health - health check with x402 status
 */
app.get("/health", (_req: Request, res: Response) => {
  const baseURL = process.env.OPENAI_BASE_URL || process.env.LLM_BASE_URL || "https://api.openai.com/v1";
  res.json({
    status: "ok",
    server: "Alith LazAI Inference Server with x402",
    host: process.env.HOST || "localhost",
    port: process.env.PORT || "8000",
    settlement: process.env.SETTLEMENT === "true",
    x402Enabled: X402_ENABLED,
    x402PaymentAmount: PAYMENT_AMOUNT,
    x402PaymentCurrency: PAYMENT_CURRENCY,
    x402Facilitator: process.env.X402_FACILITATOR_URL || "https://facilitator.aeon.xyz",
    llmBaseUrl: baseURL,
  });
});

/**
 * Run the inference server with x402 payment support
 */
export function run(
  host: string = "localhost",
  port: number = 8000,
  settlement: boolean = false,
  x402Enabled: boolean = true
): void {
  // Add LazAI settlement validation if enabled
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
    console.log(`🚀 Alith LazAI Inference Server with x402 running on http://${host}:${port}`);
    console.log(`📝 Wallet: ${client.getWallet().address}`);
    console.log(`⚙️  Settlement validation: ${settlement ? "enabled" : "disabled"}`);
    console.log(`💰 x402 Payments: ${x402Enabled ? "enabled" : "disabled"}`);
    if (x402Enabled) {
      console.log(`   Payment Amount: ${PAYMENT_AMOUNT} ${PAYMENT_CURRENCY}`);
      console.log(`   Facilitator: ${process.env.X402_FACILITATOR_URL || "https://facilitator.aeon.xyz"}`);
    }
  });
}

// Start the server if this file is run directly
if (require.main === module) {
  const PORT = parseInt(process.env.PORT || "8000");
  const HOST = process.env.HOST || "localhost";
  const SETTLEMENT = process.env.SETTLEMENT === "true";
  const X402_ENABLED_ENV = process.env.X402_ENABLED !== "false";
  
  console.log("🚀 Starting Alith LazAI Inference Server with AEON BNB x402...");
  console.log(`📝 Wallet: ${client.getWallet().address}`);
  console.log(`⚙️  Settlement validation: ${SETTLEMENT ? "enabled" : "disabled"}`);
  console.log(`💰 x402 Payments: ${X402_ENABLED_ENV ? "enabled" : "disabled"}`);
  
  run(HOST, PORT, SETTLEMENT, X402_ENABLED_ENV);
}

