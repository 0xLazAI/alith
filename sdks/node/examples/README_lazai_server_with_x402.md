# LazAI Server with AEON BNB x402 Payment Integration

This example (`lazai_server_with_x402.ts`) shows how to integrate **AEON BNB x402 HTTP 402 payment flow** with the LazAI inference server so each AI request can require an on-chain payment (USDT on BNB Chain) before processing.

---
## 1. Overview
**Features:**
- HTTP 402 Payment Required challenges with `X-402-*` headers
- Optional verification of payment proofs (placeholder in example – replace with real SDK)
- Works with Groq (LLM) via `LLM_API_KEY` / `LLM_BASE_URL`
- Supports toggling settlement validation (LazAI internal token gating) via `SETTLEMENT`
- Easily extensible to other OpenAI-compatible providers

**Key File:** `sdks/node/examples/lazai_server_with_x402.ts`

---
## 2. Prerequisites
Install dependencies (from `sdks/node` directory):

```powershell
npm install
npm install @aeon-project/bnb-x402 axios express cors dotenv
```

If you use Groq:
- Create an API key at https://console.groq.com

Optional (for real payment verification – placeholder now):
- Read AEON x402 docs: https://facilitator.aeon.xyz & https://github.com/AEON-Project/bnb-x402

---
## 3. Required .env Variables
Create or edit `sdks/node/.env`:

```env
# Wallet private key (test wallet). MUST start with 0x
PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE

# LLM provider (Groq recommended)
LLM_API_KEY=gsk_your_groq_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1

# Server network config
HOST=localhost
PORT=8000

# LazAI settlement validation (disable for simpler local testing)
SETTLEMENT=false

# x402 configuration
X402_ENABLED=true
X402_FACILITATOR_URL=https://facilitator.aeon.xyz
X402_PAYMENT_AMOUNT=0.01
X402_PAYMENT_CURRENCY=USDT

# Optional BNB Chain RPC (if you later implement direct chain reads)
BNB_CHAIN_RPC=https://bsc-dataseed.binance.org/
```

**Minimum required:** `PRIVATE_KEY`, (`LLM_API_KEY` or `OPENAI_API_KEY`), `LLM_BASE_URL` (if not using OpenAI default).

---
## 4. Running the Server
From repo root or inside `sdks/node` folder (Windows PowerShell examples):

```powershell
cd Alith\sdks\node
npx ts-node examples/lazai_server_with_x402.ts
```

Bash/macOS/Linux equivalent:
```bash
cd ./sdks/node
npx ts-node examples/lazai_server_with_x402.ts
```

You should see logs:
```
🚀 Starting Alith LazAI Inference Server with AEON BNB x402...
📝 Wallet: 0x...YourWallet
⚙️  Settlement validation: disabled
💰 x402 Payments: enabled
   Payment Amount: 0.01 USDT
   Facilitator: https://facilitator.aeon.xyz
🚀 Alith LazAI Inference Server with x402 running on http://localhost:8000
```

---
## 5. Available Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status + x402 config |
| GET | `/v1/models` | Lists available LLM models from provider |
| POST | `/v1/chat/completions` | Chat API (protected by x402 payment) |
| POST | `/v1/embeddings` | Embeddings API (protected by x402 payment) |

**HTTP 402 Behavior:** If `X402_ENABLED=true` and you omit `X-402-Payment-Proof`, server returns:
```json
{
  "error": {
    "type": "payment_required",
    "message": "Payment required: 0.01 USDT",
    "amount": "0.01",
    "currency": "USDT",
    "facilitator": "https://facilitator.aeon.xyz"
  }
}
```
Headers included:
```
X-402-Payment-Required: true
X-402-Amount: 0.01
X-402-Currency: USDT
X-402-Facilitator: https://facilitator.aeon.xyz
```

---
## 6. Payment Flow (Manual Testing)
### Step 1: Get 402 Challenge
```powershell
$body = @{ model = "llama-3.1-8b-instant"; messages = @(@{role="user";content="Hello"}) } | ConvertTo-Json
try {
  Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method POST -Body $body -ContentType "application/json"
} catch {
  $_.ErrorDetails.Message | ConvertFrom-Json | ConvertTo-Json -Depth 10
}
```

### Step 2: Create Payment Proof (Pseudo)
Replace placeholder with actual AEON SDK:
```typescript
import { X402Client } from '@aeon-project/bnb-x402';
const x402 = new X402Client({ facilitatorUrl: process.env.X402_FACILITATOR_URL });
const proof = await x402.createPayment({
  amount: process.env.X402_PAYMENT_AMOUNT,
  currency: process.env.X402_PAYMENT_CURRENCY,
  recipient: '0xServerWalletAddress'
});
```

### Step 3: Retry With Proof
```powershell
$headers = @{ "Content-Type" = "application/json"; "X-402-Payment-Proof" = "<BASE64_OR_JSON_PROOF>" }
$body = @{ model = "llama-3.1-8b-instant"; messages = @(@{role="user";content="Hello again"}) } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method POST -Body $body -Headers $headers
```

### curl Version:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"Hello"}]}'

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-402-Payment-Proof: <BASE64_OR_JSON_PROOF>" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"Hello again"}]}'
```

---
## 7. Browser Testing Page
A visual interactive test page exists at:
```
sdks/node/examples/x402-test-page.html
```
Open it in your browser (double-click in Explorer or `Start-Process` in PowerShell):
```powershell
Start-Process "C:\Users\abish\OneDrive\Desktop\Alith_forked\Alith\sdks\node\examples\x402-test-page.html"
```
Buttons available:
- Health check
- Payment required demo (402)
- Model listing
- x402 info endpoints
- Chat test (mock payment)

---
## 8. Example: Running Side-by-Side Settlement + Payment
To enable LazAI settlement validation AND x402 payment:
```env
SETTLEMENT=true
X402_ENABLED=true
```
Each request must pass both:
- Settlement headers (`X-LazAI-User`, `X-LazAI-Signature`, etc. – see core LazAI docs)
- `X-402-Payment-Proof`

---
## 9. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Settlement enabled without proper headers | Set `SETTLEMENT=false` for local quick tests |
| 402 payment_required loop | Invalid or missing payment proof | Generate real proof via AEON SDK |
| ECONNREFUSED | Server not running or wrong port | Confirm `PORT=8000` and server log shows running |
| Missing model error | Bad model name | Use `llama-3.1-8b-instant` (Groq) or list with `/v1/models` |
| PRIVATE_KEY error | .env missing or no 0x prefix | Ensure `PRIVATE_KEY=0x...` | 

---
## 10. Replacing Placeholder x402Client
In this example file a placeholder `X402Client` class simulates verification. Replace it with real SDK:
```typescript
import { X402Client } from '@aeon-project/bnb-x402';
// Initialize
const x402Client = new X402Client(process.env.X402_FACILITATOR_URL);
// Use x402Client.verifyPayment(paymentProof)
```
**Real verify logic** should return payment validity, recipient, amount, and currency.

---
## 11. Minimal Quickstart (Copy/Paste)
```powershell
# 1. Set env
Set-Content .env @'
PRIVATE_KEY=0xYOUR_PRIVATE_KEY
LLM_API_KEY=gsk_your_groq_key
LLM_BASE_URL=https://api.groq.com/openai/v1
HOST=localhost
PORT=8000
SETTLEMENT=false
X402_ENABLED=true
X402_PAYMENT_AMOUNT=0.01
X402_PAYMENT_CURRENCY=USDT
X402_FACILITATOR_URL=https://facilitator.aeon.xyz
'@

# 2. Install
npm install

# 3. Run
npx ts-node examples/lazai_server_with_x402.ts

# 4. Test (should get 402)
curl -X POST http://localhost:8000/v1/chat/completions \ 
  -H "Content-Type: application/json" \ 
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"Hello"}]}'
```

---
## 12. Next Steps
- Integrate real AEON SDK verification
- Persist successful payment metadata
- Add rate limiting per wallet
- Support multiple pricing tiers per endpoint
- Add automated integration tests

---
**Author:** Generated helper README for `lazai_server_with_x402.ts` example.

Happy building! 🔐💰🤖
