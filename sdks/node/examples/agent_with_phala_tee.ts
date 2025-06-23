/* For the deplopment environment, set the environment variable `DSTACK_SIMULATOR_ENDPOINT` with the
 * simulator: https://github.com/Leechael/tappd-simulator/releases
 *
 * In production environments, mount the socket file in your docker container:
 * ```yaml
 * volumes:
 *   - /var/run/tappd.sock:/var/run/tappd.sock
 * ```
 */
import { TEEClient, type TEEConfig } from "alith/tee/phala";

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
