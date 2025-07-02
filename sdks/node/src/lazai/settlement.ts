import Web3 from "web3";

import {
  USER_HEADER,
  NONCE_HEADER,
  SIGNATURE_HEADER,
  FILE_ID_HEADER,
} from "./request";

export class SettlementSignature {
  constructor(
    public user: string,
    public nonce: BigInt,
    public signature: string,
    public file_id?: BigInt | null
  ) {}

  toRequestHeaders(): Record<string, string> {
    return {
      [USER_HEADER]: this.user,
      [NONCE_HEADER]: this.nonce.toString(),
      [SIGNATURE_HEADER]: this.signature,
      [FILE_ID_HEADER]:
        this.file_id !== undefined && this.file_id !== null
          ? this.file_id.toString()
          : "",
    };
  }
}

export class SettlementRequest {
  constructor(
    public nonce: BigInt,
    public user: string,
    public node: string,
    public file_id?: BigInt | null
  ) {}

  abiEncode() {
    const web3 = new Web3();
    return web3.eth.abi.encodeParameters(
      ["uint256", "address", "address"],
      [this.nonce, this.user, this.node]
    );
  }

  generateSignature(privateKey: string): SettlementSignature {
    const messageHash = Web3.utils.keccak256(this.abiEncode());
    const web3 = new Web3();
    const signedMessage = web3.eth.accounts.sign(messageHash, privateKey);
    return new SettlementSignature(
      this.user,
      this.nonce,
      signedMessage.signature,
      this.file_id
    );
  }
}
