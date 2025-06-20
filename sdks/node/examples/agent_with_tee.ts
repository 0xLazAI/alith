import { Agent, TEEAgent, TEEClient, type TEEConfig } from "alith";

/**
 * Example demonstrating Phala TEE integration with Alith agents
 * This shows how to create secure AI agents with hardware-level security guarantees
 */

async function main() {
  console.log("🔒 Phala TEE Integration Example for Alith Agents");
  console.log("--------------------------------------------------");

  // 1. Basic TEE Client Usage
  console.log("\n📋 1. Setting up TEE Client...");

  const teeConfig: TEEConfig = {
    endpoint: "http://localhost:8090", // Phala TEE endpoint
    enableAttestation: true,
    enableKeyDerivation: true,
    enableSignatures: true,
    timeout: 30000,
  };

  const teeClient = new TEEClient(teeConfig);

  try {
    // Check TEE status
    const status = await teeClient.getStatus();
    console.log("TEE Status:", status);

    // Generate attestation proof
    console.log("\n🔐 2. Generating TEE Attestation...");
    const attestation = await teeClient.generateAttestation("alith-demo");
    console.log("Attestation generated:", {
      verified: attestation.verified,
      timestamp: attestation.timestamp,
      quoteLength: attestation.quote.length,
    });

    // Derive a secure key
    console.log("\n🔑 3. Deriving TEE Key...");
    const derivedKey = await teeClient.deriveKey(
      "/demo/agent",
      "alith-agent-key"
    );
    console.log("Derived key:", derivedKey);
  } catch (error) {
    console.log(
      "⚠️  TEE not available (likely running without Phala TEE environment)"
    );
    console.log("Error:", error.message);
    console.log(
      "📝 To test with real TEE, run this in a Phala TEE environment"
    );
  }

  // 2. TEE-Enabled Agent
  console.log("\n🤖 4. Creating TEE-Enabled Agent...");

  const agent = new Agent({
    name: "SecureAnalyticsAgent",
    model: "gpt-4",
    preamble:
      "You are a secure AI agent running in a Trusted Execution Environment. All your operations are cryptographically verified.",
    teeConfig: teeConfig,
  });

  // Check if TEE is enabled
  console.log("TEE Enabled:", agent.isTEEEnabled());

  // Regular prompt (automatically uses TEE if configured)
  console.log("\n💬 5. Running Secure Agent Prompt...");
  try {
    const response = await agent.prompt(
      'Analyze this sensitive financial data: {"revenue": 1000000, "costs": 750000, "profit": 250000}'
    );
    console.log("Agent Response:", response);
  } catch (error) {
    console.log(
      "Agent Response (mock):",
      "TEE-verified analysis: Financial health appears strong with 25% profit margin. Data processed securely."
    );
  }

  // Explicit secure prompt with full TEE verification
  console.log("\n🛡️  6. Running Explicit Secure Prompt...");
  try {
    const secureResult = await agent.promptSecure(
      "Perform privacy-preserving analysis on customer behavior data",
      {
        attestUserData: "customer-behavior-analysis",
        signResult: true,
        includeAttestation: true,
      }
    );

    console.log("Secure Result:", {
      verified: secureResult.verified,
      hasSignature: !!secureResult.signature,
      hasAttestation: !!secureResult.attestation,
      resultLength: secureResult.result.length,
    });
  } catch (error) {
    console.log("Secure execution would work in real TEE environment");
  }

  // 3. Standalone TEE Agent
  console.log("\n🎯 7. Using Standalone TEE Agent...");

  const teeAgent = new TEEAgent("privacy-agent", teeConfig);

  try {
    // Execute secure operation
    const secureOperation = await teeAgent.executeSecurely(async () => {
      // This runs with full TEE protection
      const sensitiveData = {
        userIds: ["user1", "user2", "user3"],
        behaviorPatterns: "anonymized-patterns",
        insights: "privacy-preserving-insights",
      };

      // Simulate ML inference within TEE
      await new Promise((resolve) => setTimeout(resolve, 100));

      return {
        processedData: "anonymized-results",
        confidence: 0.95,
        privacy: "fully-preserved",
        timestamp: new Date(),
      };
    });

    console.log("TEE Operation Result:", {
      verified: secureOperation.verified,
      hasSignature: !!secureOperation.signature,
      hasAttestation: !!secureOperation.attestation,
      result: secureOperation.result,
    });
  } catch (error) {
    console.log("TEE operation demo (would work in real environment)");
  }

  // 4. Secure Communication Channel
  console.log("\n📡 8. Creating Secure Communication Channel...");

  try {
    const channel = await teeClient.createSecureChannel();

    console.log("Secure Channel Created:", channel);

    // Test encryption/decryption
    const message = "This is a secret message protected by TEE";
    const encrypted = channel.encrypt(message);
    const decrypted = channel.decrypt(encrypted);

    console.log("Encryption Test:", {
      original: message,
      encrypted: `${encrypted.slice(0, 50)}...`,
      decrypted: decrypted,
      success: message === decrypted,
    });
  } catch (error) {
    console.log("Secure channel demo (would work in real TEE environment)");
  }

  console.log("\n✅ TEE Integration Example Complete!");
  console.log("\n📋 Summary of TEE Features:");
  console.log("   • Hardware-verified execution environment");
  console.log("   • Remote attestation with cryptographic proofs");
  console.log("   • Secure key derivation within TEE");
  console.log("   • Encrypted communication channels");
  console.log("   • Tamper-proof AI operation results");
  console.log("   • End-to-end privacy preservation");

  console.log("\n🚀 To use with real TEE:");
  console.log("   1. Deploy to Phala Network or compatible TEE environment");
  console.log("   2. Configure endpoint to your TEE instance");
  console.log("   3. All operations will have hardware security guarantees");
}

await main();
