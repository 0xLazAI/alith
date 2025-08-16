import { Mutex } from "async-mutex";
import { Web3, type Web3Account } from "web3";

export const DEVNET_NETWORK = "LazAI Devnet";
export const TESTNET_NETWORK = "LazAI Testnet";
export const LOCAL_CHAIN_ENDPOINT = "http://localhost:8545";
export const TESTNET_ENDPOINT = "https://testnet.lazai.network";
export const TESTNET_CHAINID = 133718;

export class ChainConfig {
  network: string;
  endpoint: string;
  chainId: number;
  gasMultiplier: number;
  maxRetries: number;

  constructor(network: string, endpoint: string, chainId: number) {
    this.network = network;
    this.endpoint = endpoint;
    this.chainId = chainId;
    this.gasMultiplier = 1.5;
    this.maxRetries = 3;
  }

  static local(): ChainConfig {
    return new ChainConfig(
      DEVNET_NETWORK,
      LOCAL_CHAIN_ENDPOINT,
      TESTNET_CHAINID
    );
  }

  static testnet(): ChainConfig {
    return new ChainConfig(TESTNET_NETWORK, TESTNET_ENDPOINT, TESTNET_CHAINID);
  }
}

export class ChainManager {
  config: ChainConfig;
  web3: Web3;
  account: Web3Account;
  nonceMutex: Mutex = new Mutex();

  constructor(
    config: ChainConfig = ChainConfig.testnet(),
    privateKey: string = process.env.PRIVATE_KEY || ""
  ) {
    this.config = config;
    this.web3 = new Web3(config.endpoint);
    this.account = this.web3.eth.accounts.privateKeyToAccount(privateKey);
  }

  async getCurrentBlock() {
    return this.web3.eth.getBlockNumber();
  }

  async getBalance(address?: string) {
    return this.web3.eth.getBalance(address || this.account.address);
  }

  async getNonce(address?: string) {
    return this.web3.eth.getTransactionCount(
      address || this.account.address,
      "pending"
    );
  }

  async getGasPrice() {
    return this.web3.eth.getGasPrice();
  }

  async sendTransaction(
    contractMethod: any,
    to: string,
    value: number | string = 0,
    maxRetries = 3
  ) {
    return this.nonceMutex.runExclusive(async () => {
      let retries = 0;
      while (retries < maxRetries) {
        try {
          const nonce = await this.getNonce();
          const gasEstimate: bigint = await contractMethod.estimateGas({
            from: this.account.address,
            value,
          });
          const gasPrice = await this.getGasPrice();
          const tx = {
            from: this.account.address,
            to: to,
            data: contractMethod.encodeABI(),
            gas: Math.round(Number(gasEstimate) * 1.2),
            gasPrice,
            nonce,
            value,
            chainId: this.config.chainId,
          };

          const signedTx = await this.account.signTransaction(tx);
          const receipt = await this.web3.eth.sendSignedTransaction(
            signedTx.rawTransaction
          );

          return {
            transactionHash: receipt.transactionHash,
            receipt,
          };
        } catch (error) {
          retries++;
          if (retries >= maxRetries) throw error;
          await new Promise((resolve) => setTimeout(resolve, 1000 * retries));
        }
      }
      throw new Error("Transaction failed after maximum retries");
    });
  }
}
