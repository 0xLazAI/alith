import Web3 from "web3";

export const USER_HEADER = "X-LazAI-User";
export const NONCE_HEADER = "X-LazAI-Nonce";
export const SIGNATURE_HEADER = "X-LazAI-Signature";
export const TOKEN_ID_HEADER = "X-LazAI-Token-ID";
export const FILE_ID_HEADER = "X-LazAI-File-ID";

export const QUERY_TYPE = 0;
export const INFERENCE_TYPE = 1;
export const TRAINING_TYPE = 2;

import { Client } from "./client";

export async function validateRequest(
  headers: Record<string, string>,
  type: number = QUERY_TYPE,
  client?: Client
) {
  const user = headers[USER_HEADER]!;
  const nonce = headers[NONCE_HEADER]!;
  const signature = headers[SIGNATURE_HEADER]!;
  await validateAccountAndSignature(
    user,
    parseInt(nonce),
    signature,
    type,
    client
  );
}

export async function validateAccountAndSignature(
  user: string,
  nonce: number,
  signature: string,
  type: number = TRAINING_TYPE,
  client?: Client
) {
  const clientInstance = client || new Client();
  const node = clientInstance.getWallet().address;
  let account = null;

  if (type === TRAINING_TYPE) {
    account = await clientInstance.getTrainingAccount(user, node);
  } else if (type === INFERENCE_TYPE) {
    account = await clientInstance.getInferenceAccount(user, node);
  } else {
    account = await clientInstance.getQueryAccount(user, node);
  }

  if (!account || account.node !== user) {
    throw new Error(`Account ${user} does not exist or is unauthorized`);
  }

  const lastNonce = account.nonce;
  if (nonce <= lastNonce) {
    throw new Error(
      `Invalid nonce: ${nonce}. Must be greater than last nonce: ${lastNonce}`
    );
  }

  const recoveredAddress = recoverAddress(nonce, user, node, signature);
  if (recoveredAddress.toLowerCase() !== user.toLowerCase()) {
    throw new Error("Signature verification failed: address mismatch");
  }
}

export function recoverAddress(
  nonce: number,
  user: string,
  node: string,
  signature: string
): string {
  const web3 = new Web3();
  const encodedData = web3.eth.abi.encodeParameter(
    "(uint256, address, address)",
    [nonce, user, node]
  );
  const messageHash = Web3.utils.keccak256(encodedData);
  const recoveredAddress = web3.eth.accounts.recover(messageHash, signature);
  return recoveredAddress;
}
