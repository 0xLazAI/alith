/*
 * Alith Marlin TEE Integration & SDK. This SDK provides a Rust client for communicating with the attestation server.
 *
 * For local development and testing without TDX devices, you can use the simulator available for download here:
 * https://github.com/marlinprotocol/oyster-monorepo/tree/master/attestation/server-custom-mock and then set the
 * environment variable `MARLIN_ATTESTATION_ENDPOINT` (Optional, default is http://127.0.0.1:1350)
 *
 * # From Source
 * ```no_check
 * git clone https://github.com/marlinprotocol/oyster-monorepo
 * cd oyster-monorepo/attestation/server-custom-mock
 *
 * # Listens on 127.0.0.1:1350 by default
 * cargo run -r
 *
 * # To customize listening interface and port
 * cargo run -r --ip-addr <ip>:<port>
 * ```
 * # From Docker
 * ```no_check
 * # The server runs on 1350 inside Docker, can remap to any interface and port
 * docker run --init -p http://127.0.0.1:1350 marlinorg/attestation-server-custom-mock
 * ```
 */

import axios from "axios";

// Default endpoint for Marlin attestation server
export const DEFAULT_MARLIN_ATTESTATION_ENDPOINT = "http://127.0.0.1:1350";
export const MARLIN_ATTESTATION_ENDPOINT_ENV = "MARLIN_ATTESTATION_ENDPOINT";

/**
 * Comprehensive error enumeration for Marlin client operations.
 * Encapsulates HTTP and JSON parsing errors.
 */
export class MarlinError extends Error {
  constructor(
    public readonly type: "HttpError" | "JsonError",
    message: string
  ) {
    super(message);
    this.name = "MarlinError";
  }
}

/**
 * Convenience type alias for Marlin client results.
 */
export type Result<T> = Promise<
  { success: true; data: T } | { success: false; error: MarlinError }
>;

/**
 * Attestation request structure.
 * Contains optional public key, user data, and nonce.
 */
export class AttestationRequest {
  publicKey?: Uint8Array;
  userData?: Uint8Array;
  nonce?: Uint8Array;

  constructor(
    publicKey?: Uint8Array,
    userData?: Uint8Array,
    nonce?: Uint8Array
  ) {
    this.publicKey = publicKey;
    this.userData = userData;
    this.nonce = nonce;
  }
}

/**
 * Main Marlin client class.
 * Manages connections to Marlin services and provides methods for common operations.
 */
export class MarlinClient {
  private client = axios.create();
  private endpoint: string;

  /**
   * Create a new Marlin client instance.
   *
   * Automatically selects connection method based on endpoint:
   * - HTTP/HTTPS URLs: Standard network connection
   * - Default behavior (no endpoint specified):
   *   1. Check environment variable `MARLIN_ATTESTATION_ENDPOINT`
   *   2. Fall back to `http://127.0.0.1:1350`
   */
  constructor(endpoint?: string) {
    this.endpoint = getEndpoint(endpoint);
  }

  /**
   * Generate a remote attestation with the public key and user data.
   * @param req Attestation request containing public key, user data, and nonce
   * @returns Promise resolving to the attestation result
   */
  public async attestationHex(req: AttestationRequest): Promise<string> {
    try {
      // Convert Uint8Array to hex strings, handling undefined values
      const publicKeyHex = req.publicKey ? uint8ArrayToHex(req.publicKey) : "";
      const userDataHex = req.userData ? uint8ArrayToHex(req.userData) : "";
      const nonceHex = req.nonce ? uint8ArrayToHex(req.nonce) : "";

      // Construct request URL
      const url =
        `${this.endpoint}/attestation/hex?` +
        `public_key=${publicKeyHex}&` +
        `user_data=${userDataHex}&` +
        `nonce=${nonceHex}`;

      // Send GET request and handle response
      const response = await this.client.get(url);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new MarlinError("HttpError", `HTTP error: ${error.message}`);
      }
      throw new MarlinError(
        "JsonError",
        `JSON parsing error: ${(error as Error).message}`
      );
    }
  }
}

/**
 * Resolve the endpoint URL based on provided parameter, environment variable, or default.
 * @param endpoint Optional endpoint parameter
 * @returns Resolved endpoint URL
 */
function getEndpoint(endpoint?: string): string {
  return (
    endpoint ||
    process.env[MARLIN_ATTESTATION_ENDPOINT_ENV] ||
    DEFAULT_MARLIN_ATTESTATION_ENDPOINT
  );
}

/**
 * Convert Uint8Array to hexadecimal string.
 * @param data Byte array to convert
 * @returns Hexadecimal string representation
 */
function uint8ArrayToHex(data: Uint8Array): string {
  return Array.from(data)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

// Export MarlinClient as default
export default MarlinClient;
