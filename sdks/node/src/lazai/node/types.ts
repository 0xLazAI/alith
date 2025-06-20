export interface ProofRequest {
  jobId: number;
  fileId: number;
  fileUrl: string;
  encryptedKey: string;
  proofUrl: string | null;
  encryption_seed: string | null;
  nonce: number | null;
}
