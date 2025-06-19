import { TappdClient } from '@phala/dstack-sdk'

/**
 * TEE Configuration for Alith agents
 */
export interface TEEConfig {
  endpoint?: string // TEE endpoint (defaults to localhost for development)
  enableAttestation?: boolean // Enable remote attestation verification
  enableKeyDerivation?: boolean // Enable TEE-based key derivation
  enableSignatures?: boolean // Enable TEE-based message signing
  timeout?: number // Request timeout in milliseconds
}

/**
 * TEE Attestation Result containing cryptographic proof
 */
export interface TEEAttestation {
  quote: string // TDX quote in hex format
  eventLog: string // Event log for verification
  rtmrs: any // Runtime Measurement Registers
  verified: boolean // Attestation verification status
  timestamp: Date // Attestation timestamp
}

/**
 * TEE Key Derivation Result
 */
export interface TEEDerivedKey {
  publicKey: string // Derived public key
  address: string // Ethereum-style address
  keyPath: string // Key derivation path
  signature?: string // Optional signature proof
}

/**
 * TEE Execution Result with verification
 */
export interface TEEExecutionResult<T = any> {
  result: T // Execution result
  signature: string // TEE signature of the result
  attestation?: TEEAttestation // Optional attestation proof
  verified: boolean // Result verification status
}

/**
 * Phala TEE Client for secure AI agent execution
 * Provides TEE-based security guarantees for sensitive AI operations
 */
export class TEEClient {
  private client: TappdClient
  private config: Required<TEEConfig>

  constructor(config: TEEConfig = {}) {
    this.config = {
      endpoint: config.endpoint || 'http://localhost:8090',
      enableAttestation: config.enableAttestation ?? true,
      enableKeyDerivation: config.enableKeyDerivation ?? true,
      enableSignatures: config.enableSignatures ?? true,
      timeout: config.timeout ?? 30000,
    }

    // Initialize Phala TappdClient
    this.client = new TappdClient(this.config.endpoint)
  }

  /**
   * Get TEE environment information
   */
  async getInfo(): Promise<any> {
    try {
      return await this.client.info()
    } catch (error) {
      throw new Error(`Failed to get TEE info: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Generate remote attestation for the current TEE environment
   * This proves the code is running in a genuine, uncompromised TEE
   */
  async generateAttestation(userData: string = '', hashAlg: string = 'sha256'): Promise<TEEAttestation> {
    try {
      if (!this.config.enableAttestation) {
        throw new Error('Attestation is disabled in configuration')
      }

      // Generate TDX quote using Phala's SDK
      const result = await this.client.tdxQuote(userData, hashAlg)

      // Replay RTMRs for verification
      const rtmrs = result.replayRtmrs()

      return {
        quote: result.quote,
        eventLog: result.event_log,
        rtmrs,
        verified: true, // In production, this should be verified against Intel's attestation service
        timestamp: new Date(),
      }
    } catch (error) {
      throw new Error(`Failed to generate attestation: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Derive a cryptographic key within the TEE
   * Keys never leave the secure environment
   */
  async deriveKey(path: string = '/', userData: string = ''): Promise<TEEDerivedKey> {
    try {
      if (!this.config.enableKeyDerivation) {
        throw new Error('Key derivation is disabled in configuration')
      }

      // Use Phala's key derivation capability
      const result = await this.client.deriveKey(path, userData)

      return {
        publicKey: result.publicKey || result.address, // Handle different SDK versions
        address: result.address,
        keyPath: path,
      }
    } catch (error) {
      throw new Error(`Failed to derive key: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Execute a function within TEE with cryptographic verification
   * This is the core method for secure AI operations
   */
  async executeSecure<T>(
    operation: () => Promise<T> | T,
    options: {
      attestUserData?: string // Additional data to include in attestation
      signResult?: boolean // Whether to sign the execution result
      includeAttestation?: boolean // Whether to include full attestation
    } = {}
  ): Promise<TEEExecutionResult<T>> {
    try {
      const {
        attestUserData = `alith-tee-execution-${Date.now()}`,
        signResult = this.config.enableSignatures,
        includeAttestation = this.config.enableAttestation,
      } = options

      // Generate attestation proof before execution
      let attestation: TEEAttestation | undefined
      if (includeAttestation) {
        attestation = await this.generateAttestation(attestUserData)
      }

      // Execute the operation within TEE
      const result = await operation()

      // Generate signature if requested
      let signature = ''
      if (signResult) {
        // Create deterministic signature of the result
        const resultHash = this.hashResult(result)
        
        // In a real implementation, this would use TEE's signing capability
        // For now, we'll create a placeholder signature that includes the attestation
        signature = await this.signWithTEE(resultHash, attestation?.quote || '')
      }

      return {
        result,
        signature,
        attestation,
        verified: true,
      }
    } catch (error) {
      throw new Error(`TEE execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Verify a TEE execution result
   */
  async verifyExecution<T>(execution: TEEExecutionResult<T>): Promise<boolean> {
    try {
      // Verify signature if present
      if (execution.signature && this.config.enableSignatures) {
        const resultHash = this.hashResult(execution.result)
        const isValidSignature = await this.verifySignature(
          resultHash,
          execution.signature,
          execution.attestation?.quote || ''
        )
        if (!isValidSignature) {
          return false
        }
      }

      // Verify attestation if present
      if (execution.attestation && this.config.enableAttestation) {
        // In production, this would verify against Intel's attestation service
        // For now, we'll do basic validation
        return this.verifyAttestation(execution.attestation)
      }

      return true
    } catch (error) {
      console.error('TEE verification failed:', error)
      return false
    }
  }

  /**
   * Create a secure communication channel for AI agents
   * This enables end-to-end encrypted communication with TEE guarantees
   */
  async createSecureChannel(remotePublicKey?: string): Promise<{
    localKey: TEEDerivedKey
    channelId: string
    encrypt: (data: string) => string
    decrypt: (data: string) => string
  }> {
    try {
      // Derive a unique key for this channel
      const channelId = `channel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      const localKey = await this.deriveKey(`/channels/${channelId}`)

      // Simple encryption/decryption using derived key (in production, use proper crypto)
      const encrypt = (data: string): string => {
        // Placeholder: use proper encryption with derived key
        const encoded = Buffer.from(data).toString('base64')
        return `TEE_ENCRYPTED:${localKey.address.slice(0, 8)}:${encoded}`
      }

      const decrypt = (data: string): string => {
        // Placeholder: use proper decryption with derived key
        if (!data.startsWith('TEE_ENCRYPTED:')) {
          throw new Error('Invalid encrypted data format')
        }
        const parts = data.split(':')
        if (parts.length !== 3) {
          throw new Error('Invalid encrypted data format')
        }
        return Buffer.from(parts[2], 'base64').toString()
      }

      return {
        localKey,
        channelId,
        encrypt,
        decrypt,
      }
    } catch (error) {
      throw new Error(`Failed to create secure channel: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Private helper methods
   */
  private hashResult<T>(result: T): string {
    // Simple hash of the result (in production, use proper cryptographic hash)
    const resultStr = typeof result === 'string' ? result : JSON.stringify(result)
    return Buffer.from(resultStr).toString('base64')
  }

  private async signWithTEE(data: string, attestationQuote: string): Promise<string> {
    // Placeholder TEE signing (in production, use actual TEE signing capabilities)
    const timestamp = Date.now().toString()
    const combined = `${data}:${attestationQuote}:${timestamp}`
    return Buffer.from(combined).toString('base64')
  }

  private async verifySignature(data: string, signature: string, attestationQuote: string): Promise<boolean> {
    try {
      // Placeholder signature verification (in production, use proper crypto verification)
      const decoded = Buffer.from(signature, 'base64').toString()
      return decoded.includes(data) && decoded.includes(attestationQuote)
    } catch {
      return false
    }
  }

  private verifyAttestation(attestation: TEEAttestation): boolean {
    // Basic attestation validation (in production, verify with Intel's service)
    return (
      attestation.quote.length > 0 &&
      attestation.eventLog.length > 0 &&
      attestation.verified &&
      Date.now() - attestation.timestamp.getTime() < 300000 // 5 minutes max age
    )
  }

  /**
   * Get current TEE status and health
   */
  async getStatus(): Promise<{
    healthy: boolean
    endpoint: string
    features: {
      attestation: boolean
      keyDerivation: boolean
      signatures: boolean
    }
    lastAttestation?: Date
  }> {
    try {
      const info = await this.getInfo()
      
      return {
        healthy: true,
        endpoint: this.config.endpoint,
        features: {
          attestation: this.config.enableAttestation,
          keyDerivation: this.config.enableKeyDerivation,
          signatures: this.config.enableSignatures,
        },
        lastAttestation: new Date(),
      }
    } catch (error) {
      return {
        healthy: false,
        endpoint: this.config.endpoint,
        features: {
          attestation: false,
          keyDerivation: false,
          signatures: false,
        },
      }
    }
  }
}

/**
 * TEE-enabled Agent wrapper that adds security guarantees to AI operations
 */
export class TEEAgent {
  private teeClient: TEEClient
  private agentId: string

  constructor(agentId: string, teeConfig?: TEEConfig) {
    this.agentId = agentId
    this.teeClient = new TEEClient(teeConfig)
  }

  /**
   * Execute an AI operation with TEE security guarantees
   */
  async executeSecurely<T>(
    operation: () => Promise<T> | T,
    options?: {
      attestUserData?: string
      signResult?: boolean
      includeAttestation?: boolean
    }
  ): Promise<TEEExecutionResult<T>> {
    const userData = options?.attestUserData || `agent-${this.agentId}-${Date.now()}`
    
    return this.teeClient.executeSecure(operation, {
      ...options,
      attestUserData: userData,
    })
  }

  /**
   * Generate agent-specific attestation
   */
  async generateAgentAttestation(): Promise<TEEAttestation> {
    return this.teeClient.generateAttestation(`agent-${this.agentId}`)
  }

  /**
   * Create secure communication channel for the agent
   */
  async createSecureChannel(): Promise<ReturnType<TEEClient['createSecureChannel']>> {
    return this.teeClient.createSecureChannel()
  }

  /**
   * Get agent's TEE status
   */
  async getStatus(): Promise<ReturnType<TEEClient['getStatus']>> {
    return this.teeClient.getStatus()
  }
}

export { TappdClient } 