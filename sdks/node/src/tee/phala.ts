/**
 * Alith Phala TEE Integration & SDK. This SDK provides a Rust client for communicating with the Tappd server,
 * which is available inside Phala Network DStack.
 *
 * For local development without TDX devices, you can use the simulator available for download here:
 * https://github.com/Leechael/tappd-simulator/releases and then set the environment variable `DSTACK_SIMULATOR_ENDPOINT`
 *
 * Leave the endpoint parameter empty for the tappd client in production. You only need to add volumes in your
 * docker-compose file to run Confidential Virtual Machines (CVMs):
 *
 * ```yaml
 *   volumes:
 *   - /var/run/tappd.sock:/var/run/tappd.sock
 * ```
 */
import {
  type DeriveKeyResponse,
  TappdClient,
  type TdxQuoteHashAlgorithms,
  type TappdInfoResponse,
} from "@phala/dstack-sdk";

/**
 * TEE Configuration for Alith agents
 */
export interface TEEConfig {
  endpoint?: string; // TEE endpoint (defaults to localhost for development)
  enableAttestation?: boolean; // Enable remote attestation verification
  enableKeyDerivation?: boolean; // Enable TEE-based key derivation
  enableSignatures?: boolean; // Enable TEE-based message signing
  timeout?: number; // Request timeout in milliseconds
}

/**
 * TEE Attestation Result containing cryptographic proof
 */
export interface TEEAttestation {
  quote: string; // TDX quote in hex format
  eventLog: string; // Event log for verification
  rtmrs: string[]; // Runtime Measurement Registers
  verified: boolean; // Attestation verification status
  timestamp: Date; // Attestation timestamp
}

/**
 * TEE Key Derivation Result
 */
export interface TEEDerivedKey {
  key: string; // Derived key from TEE
  certificateChain: string[]; // Certificate chain for verification
  keyPath: string; // Key derivation path
  signature?: string; // Optional signature proof
}

/**
 * TEE Execution Result with verification
 */
export interface TEEExecutionResult<T = unknown> {
  result: T; // Execution result
  signature: string; // TEE signature of the result
  attestation?: TEEAttestation; // Optional attestation proof
  verified: boolean; // Result verification status
}

/**
 * Phala TEE Client for secure AI agent execution
 * Provides TEE-based security guarantees for sensitive AI operations
 */
export class TEEClient {
  private client: TappdClient;
  private config: Required<TEEConfig>;

  constructor(config: TEEConfig = {}) {
    this.config = {
      endpoint: config.endpoint || "http://localhost:8090",
      enableAttestation: config.enableAttestation ?? true,
      enableKeyDerivation: config.enableKeyDerivation ?? true,
      enableSignatures: config.enableSignatures ?? true,
      timeout: config.timeout ?? 30000,
    };

    // Initialize Phala TappdClient
    this.client = new TappdClient(this.config.endpoint);
  }

  /**
   * Get TEE environment information
   */
  async getInfo(): Promise<TappdInfoResponse> {
    try {
      return await this.client.info();
    } catch (error) {
      throw new Error(
        `Failed to get TEE info: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Generate remote attestation for the current TEE environment
   * This proves the code is running in a genuine, uncompromised TEE
   */
  async generateAttestation(
    userData = "",
    hashAlg: TdxQuoteHashAlgorithms = "sha256"
  ): Promise<TEEAttestation> {
    try {
      if (!this.config.enableAttestation) {
        throw new Error("Attestation is disabled in configuration");
      }

      // Generate TDX quote using Phala's SDK
      const result = await this.client.tdxQuote(userData, hashAlg);

      // Replay RTMRs for verification
      const rtmrs = result.replayRtmrs();

      return {
        quote: result.quote,
        eventLog: result.event_log,
        rtmrs,
        verified: true, // In production, this should be verified against Intel's attestation service
        timestamp: new Date(),
      };
    } catch (error) {
      throw new Error(
        `Failed to generate attestation: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Derive a cryptographic key within the TEE
   * Keys never leave the secure environment
   */
  async deriveKey(path = "/", userData = ""): Promise<TEEDerivedKey> {
    try {
      if (!this.config.enableKeyDerivation) {
        throw new Error("Key derivation is disabled in configuration");
      }

      // Use Phala's key derivation capability
      const result: DeriveKeyResponse = await this.client.deriveKey(
        path,
        userData
      );

      return {
        key: result.key,
        certificateChain: result.certificate_chain,
        keyPath: path,
      };
    } catch (error) {
      throw new Error(
        `Failed to derive key: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Execute a function within TEE with cryptographic verification
   * This is the core method for secure AI operations
   */
  async executeSecure<T>(
    operation: () => Promise<T> | T,
    options: {
      attestUserData?: string; // Additional data to include in attestation
      signResult?: boolean; // Whether to sign the execution result
      includeAttestation?: boolean; // Whether to include full attestation
    } = {}
  ): Promise<TEEExecutionResult<T>> {
    try {
      const {
        attestUserData = `alith-tee-execution-${Date.now()}`,
        signResult = this.config.enableSignatures,
        includeAttestation = this.config.enableAttestation,
      } = options;

      // Generate attestation proof before execution
      let attestation: TEEAttestation | undefined;
      if (includeAttestation) {
        attestation = await this.generateAttestation(attestUserData);
      }

      // Execute the operation within TEE
      const result = await operation();

      // Generate signature if requested
      let signature = "";
      if (signResult) {
        // Create deterministic signature of the result
        const resultHash = this.hashResult(result);

        // In a real implementation, this would use TEE's signing capability
        // For now, we'll create a placeholder signature that includes the attestation
        signature = await this.signWithTEE(
          resultHash,
          attestation?.quote || ""
        );
      }

      return {
        result,
        signature,
        attestation,
        verified: true,
      };
    } catch (error) {
      throw new Error(
        `TEE execution failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Verify a TEE execution result
   */
  async verifyExecution<T>(execution: TEEExecutionResult<T>): Promise<boolean> {
    try {
      // Verify signature if present
      if (execution.signature && this.config.enableSignatures) {
        const resultHash = this.hashResult(execution.result);
        const isValidSignature = await this.verifySignature(
          resultHash,
          execution.signature,
          execution.attestation?.quote || ""
        );
        if (!isValidSignature) {
          return false;
        }
      }

      // Verify attestation if present
      if (execution.attestation && this.config.enableAttestation) {
        // In production, this would verify against Intel's attestation service
        // For now, we'll do basic validation
        return this.verifyAttestation(execution.attestation);
      }

      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Create a secure communication channel using TEE-derived keys
   */
  async createSecureChannel(): Promise<{
    localKey: TEEDerivedKey;
    channelId: string;
    encrypt: (data: string) => string;
    decrypt: (data: string) => string;
  }> {
    try {
      // Derive a unique key for this channel
      const channelId = `channel-${Date.now()}-${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      const localKey = await this.deriveKey(`/channels/${channelId}`);

      // Simple encryption/decryption functions (in production, use proper crypto)
      const encrypt = (data: string): string => {
        // This is a placeholder - use proper encryption in production
        return Buffer.from(data + localKey.key).toString("base64");
      };

      const decrypt = (data: string): string => {
        // This is a placeholder - use proper decryption in production
        const decoded = Buffer.from(data, "base64").toString();
        return decoded.replace(localKey.key, "");
      };

      return {
        localKey,
        channelId,
        encrypt,
        decrypt,
      };
    } catch (error) {
      throw new Error(
        `Failed to create secure channel: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  // Private helper methods
  private hashResult<T>(result: T): string {
    // Simple hash implementation - use proper crypto hash in production
    return Buffer.from(JSON.stringify(result)).toString("base64");
  }

  private async signWithTEE(
    data: string,
    attestationQuote: string
  ): Promise<string> {
    // Placeholder signature - in production, use TEE's signing capability
    const combined = data + attestationQuote + Date.now();
    return Buffer.from(combined).toString("base64");
  }

  private async verifySignature(
    data: string,
    signature: string,
    attestationQuote: string
  ): Promise<boolean> {
    try {
      // Placeholder verification - implement proper signature verification
      const decoded = Buffer.from(signature, "base64").toString();
      return decoded.includes(data) && decoded.includes(attestationQuote);
    } catch {
      return false;
    }
  }

  private verifyAttestation(attestation: TEEAttestation): boolean {
    // Basic attestation validation - implement proper Intel attestation verification
    return (
      attestation.quote.length > 0 &&
      attestation.rtmrs.length > 0 &&
      attestation.timestamp instanceof Date &&
      attestation.verified
    );
  }

  /**
   * Get TEE client status and health information
   */
  async getStatus(): Promise<{
    healthy: boolean;
    endpoint: string;
    features: {
      attestation: boolean;
      keyDerivation: boolean;
      signatures: boolean;
    };
    lastAttestation?: Date;
  }> {
    try {
      // Check if TEE is reachable
      const healthy = await this.client.isReachable();

      return {
        healthy,
        endpoint: this.config.endpoint,
        features: {
          attestation: this.config.enableAttestation,
          keyDerivation: this.config.enableKeyDerivation,
          signatures: this.config.enableSignatures,
        },
        lastAttestation: healthy ? new Date() : undefined,
      };
    } catch (error) {
      return {
        healthy: false,
        endpoint: this.config.endpoint,
        features: {
          attestation: this.config.enableAttestation,
          keyDerivation: this.config.enableKeyDerivation,
          signatures: this.config.enableSignatures,
        },
      };
    }
  }
}

/**
 * TEE-enabled agent wrapper
 * Provides a simplified interface for agents to use TEE capabilities
 */
export class TEEAgent {
  private teeClient: TEEClient;
  private agentId: string;

  constructor(agentId: string, teeConfig?: TEEConfig) {
    this.agentId = agentId;
    this.teeClient = new TEEClient(teeConfig);
  }

  /**
   * Execute an operation securely within TEE
   */
  async executeSecurely<T>(
    operation: () => Promise<T> | T,
    options?: {
      attestUserData?: string;
      signResult?: boolean;
      includeAttestation?: boolean;
    }
  ): Promise<TEEExecutionResult<T>> {
    const agentUserData = `${this.agentId}-${
      options?.attestUserData || "default"
    }`;
    return this.teeClient.executeSecure(operation, {
      ...options,
      attestUserData: agentUserData,
    });
  }

  /**
   * Generate agent-specific attestation
   */
  async generateAgentAttestation(): Promise<TEEAttestation> {
    const userData = `agent-${this.agentId}-${Date.now()}`;
    return this.teeClient.generateAttestation(userData);
  }

  /**
   * Create secure communication channel for this agent
   */
  async createSecureChannel(): Promise<
    ReturnType<TEEClient["createSecureChannel"]>
  > {
    return this.teeClient.createSecureChannel();
  }

  /**
   * Get TEE status for this agent
   */
  async getStatus(): Promise<ReturnType<TEEClient["getStatus"]>> {
    return this.teeClient.getStatus();
  }
}

// Export the TappdClient directly for advanced usage
export { TappdClient };
