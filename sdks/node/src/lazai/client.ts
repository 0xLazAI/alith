import { ChainConfig, ChainManager } from "./chain";
import {
  AI_PROCESS_CONTRACT_ABI,
  ContractConfig,
  DATA_ANCHORING_TOKEN_CONTRACT,
  DATA_REGISTRY_CONTRACT_ABI,
  SETTLEMENT_CONTRACT_ABI,
  VERIFIED_COMPUTING_CONTRACT_ABI,
} from "./contracts";
import type { ProofData, SettlementData } from "./proof";
import { randomBytes } from "crypto";
import Web3 from "web3";
import { SettlementRequest } from "./settlement";

export class Client extends ChainManager {
  contractConfig: ContractConfig;

  constructor(
    chainConfig: ChainConfig = ChainConfig.testnet(),
    contractConfig: ContractConfig = ContractConfig.testnet(),
    privateKey: string = process.env.PRIVATE_KEY || ""
  ) {
    super(chainConfig, privateKey);
    this.contractConfig = contractConfig;
  }

  dataRegistryContract() {
    return new this.web3.eth.Contract(
      DATA_REGISTRY_CONTRACT_ABI,
      this.contractConfig.dataRegistryAddress
    );
  }

  verifiedComputingContract() {
    return new this.web3.eth.Contract(
      VERIFIED_COMPUTING_CONTRACT_ABI,
      this.contractConfig.verifiedComputingAddress
    );
  }

  dataAnchoringTokenContract() {
    return new this.web3.eth.Contract(
      DATA_ANCHORING_TOKEN_CONTRACT,
      this.contractConfig.dataAnchoringTokenAddress
    );
  }

  queryContract() {
    return new this.web3.eth.Contract(
      AI_PROCESS_CONTRACT_ABI,
      this.contractConfig.queryAddress
    );
  }

  inferenceContract() {
    return new this.web3.eth.Contract(
      AI_PROCESS_CONTRACT_ABI,
      this.contractConfig.inferenceAddress
    );
  }

  trainingContract() {
    return new this.web3.eth.Contract(
      AI_PROCESS_CONTRACT_ABI,
      this.contractConfig.trainingAddress
    );
  }

  settlementContract() {
    return new this.web3.eth.Contract(
      SETTLEMENT_CONTRACT_ABI,
      this.contractConfig.settlementAddress
    );
  }

  getWallet() {
    return this.account;
  }

  async getPublicKey(): Promise<string> {
    return this.dataRegistryContract().methods.publicKey().call();
  }

  async addFile(url: string): Promise<bigint> {
    return this.addFileWithHash(url, "");
  }

  async addFileWithHash(url: string, hash: string): Promise<bigint> {
    const method = this.dataRegistryContract().methods.addFile(url, hash);
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress);
    return this.getFileIdByUrl(url);
  }

  async getFileIdByUrl(url: string): Promise<bigint> {
    return this.dataRegistryContract().methods.getFileIdByUrl(url).call();
  }

  async addNode(address: string, url: string, publicKey: string) {
    const method = this.verifiedComputingContract().methods.addNode(
      address,
      url,
      publicKey
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress
    );
  }

  async addProof(fileId: bigint, data: ProofData) {
    const messageHash = Web3.utils.keccak256(data.abiEncode());
    const signature = this.web3.eth.accounts.sign(
      messageHash,
      this.account.privateKey
    );

    const proof = {
      signature: signature.signature,
      data: {
        id: data.id,
        score: data.score,
        fileUrl: data.fileUrl,
        proofUrl: data.proofUrl,
      },
    };

    const method = this.dataRegistryContract().methods.addProof(fileId, proof);
    return await this.sendTransaction(
      method,
      this.contractConfig.dataRegistryAddress
    );
  }

  async addFileWithPermissions(
    url: string,
    ownerAddress: string,
    permissions: { account: string; key: string }[]
  ): Promise<bigint> {
    const method = this.dataRegistryContract().methods.addFileWithPermissions(
      url,
      ownerAddress,
      permissions
    );
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress);
    return this.getFileIdByUrl(url);
  }

  async addPermissionForFile(fileId: bigint, account: string, key: string) {
    const method = this.dataRegistryContract().methods.addPermissionForFile(
      fileId,
      account,
      key
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.dataRegistryAddress
    );
  }

  async getFile(fileId: bigint): Promise<{
    id: bigint;
    url: string;
    owner: string;
    createdAt: bigint;
  }> {
    return this.dataRegistryContract().methods.getFile(fileId).call();
  }

  async getFilePermission(fileId: bigint, account: string): Promise<string> {
    return this.dataRegistryContract()
      .methods.getFilePermission(fileId, account)
      .call();
  }

  async getFileProof(
    fileId: bigint,
    index: bigint
  ): Promise<{
    signature: string;
    data: {
      id: bigint;
      fileUrl: string;
      proofUrl: string;
    };
  }> {
    return this.dataRegistryContract()
      .methods.getFileProof(fileId, index)
      .call();
  }

  async getFilesCount(): Promise<bigint> {
    return this.dataRegistryContract().methods.filesCount().call();
  }

  async requestReward(fileId: bigint, proofIndex: bigint = BigInt(1)) {
    const method = this.dataRegistryContract().methods.requestReward(
      fileId,
      proofIndex
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.dataRegistryAddress
    );
  }

  async removeNode(address: string) {
    const method = this.verifiedComputingContract().methods.removeNode(address);
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress
    );
  }

  async nodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.nodeList().call();
  }

  async getNode(address: string): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.verifiedComputingContract().methods.getNode(address).call();
  }

  async updateNodeFee(fee: bigint) {
    const method = this.verifiedComputingContract().methods.updateNodeFee(fee);
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress
    );
  }

  async nodeFee(): Promise<bigint> {
    return this.verifiedComputingContract().methods.nodeFee().call();
  }

  async requestProof(fileId: bigint, value: bigint = BigInt(0)) {
    const method =
      this.verifiedComputingContract().methods.requestProof(fileId);
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress,
      value.toString()
    );
  }

  async completeJob(jobId: bigint) {
    const method = this.verifiedComputingContract().methods.completeJob(jobId);
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress
    );
  }

  async getJob(jobId: bigint): Promise<{
    fileId: bigint;
    bidAmount: string;
    status: number;
    addedTimestamp: bigint;
    ownerAddress: string;
    nodeAddress: string;
  }> {
    return this.verifiedComputingContract().methods.getJob(jobId).call();
  }

  async fileJobIds(fileId: bigint): Promise<bigint[]> {
    return this.verifiedComputingContract().methods.fileJobIds(fileId).call();
  }

  async jobsCount(): Promise<bigint> {
    return this.verifiedComputingContract().methods.jobsCount().call();
  }

  async nodeListAt(index: bigint): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.verifiedComputingContract().methods.nodeListAt(index).call();
  }

  async activeNodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.activeNodeList().call();
  }

  async activeNodeListAt(index: bigint): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.verifiedComputingContract()
      .methods.activeNodeListAt(index)
      .call();
  }

  async nodesCount(): Promise<bigint> {
    return this.verifiedComputingContract().methods.nodesCount().call();
  }

  async activeNodesCount(): Promise<bigint> {
    return this.verifiedComputingContract().methods.activeNodesCount().call();
  }

  async isNode(address: string): Promise<boolean> {
    return this.verifiedComputingContract().methods.isNode(address).call();
  }

  async submitJob(fileId: bigint, value: bigint): Promise<void> {
    const method = this.verifiedComputingContract().methods.submitJob(fileId);
    await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress,
      value.toString()
    );
  }

  async claim() {
    const method = this.verifiedComputingContract().methods.claim();
    return await this.sendTransaction(
      method,
      this.contractConfig.verifiedComputingAddress
    );
  }

  /**
   * Mint a new Data Anchor Token (DAT) with the specified parameters.
   */
  async mintDAT() {
    const method = this.dataAnchoringTokenContract().methods.mint;
    return await this.sendTransaction(
      method,
      this.contractConfig.dataAnchoringTokenAddress
    );
  }

  /**
   * Returns the balance of a specific Data Anchor Token (DAT) for a given account and token ID.
   */
  async getDATBalance(account: string, id: bigint): Promise<bigint> {
    return this.dataAnchoringTokenContract()
      .methods.balanceOf(account, id)
      .call();
  }

  /**
   * Returns the Uri for a specific Data Anchor Token (DAT) by its token ID.
   */
  async dataUri(tokenId: bigint): Promise<string> {
    return this.dataAnchoringTokenContract().methods.uri(tokenId).call();
  }

  async getUser(address: string): Promise<{
    addr: string;
    availableBalance: bigint;
    totalBalance: bigint;
    queryNodes: string[];
    inferenceNodes: string[];
    trainingNodes: string[];
  }> {
    return this.settlementContract().methods.getUser(address).call();
  }

  async getAllUsers(): Promise<
    {
      addr: string;
      availableBalance: bigint;
      totalBalance: bigint;
      inferenceNodes: string[];
      trainingNodes: string[];
    }[]
  > {
    return this.settlementContract().methods.getAllUser().call();
  }

  async addUser(amount: number | string) {
    const method = this.settlementContract().methods.addUser();
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress,
      amount
    );
  }

  async deleteUser() {
    const method = this.settlementContract().methods.deleteUser();
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async deposit(amount: number | string) {
    const method = this.settlementContract().methods.deposit();
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress,
      amount
    );
  }

  async withdraw(amount: number | string) {
    const method = this.settlementContract().methods.withdraw(amount);
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async depositQuery(node: string, amount: number) {
    const method = this.settlementContract().methods.depositQuery(node, amount);
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async depositInference(node: string, amount: number) {
    const method = this.settlementContract().methods.depositInference(
      node,
      amount
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async depositTraining(node: string, amount: number) {
    const method = this.settlementContract().methods.depositTraining(
      node,
      amount
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async retrieveQuery(nodes: string[]) {
    const method = this.settlementContract().methods.retrieveQuery(nodes);
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async retrieveInference(nodes: string[]) {
    const method = this.settlementContract().methods.retrieveInference(nodes);
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async retrieveTraining(nodes: string[]) {
    const method = this.settlementContract().methods.retrieveTraining(nodes);
    return await this.sendTransaction(
      method,
      this.contractConfig.settlementAddress
    );
  }

  async addQueryNode(address: string, url: string, public_key: string) {
    const method = this.queryContract().methods.addNode(
      address,
      url,
      public_key
    );
    return await this.sendTransaction(method, this.contractConfig.queryAddress);
  }

  async removeQueryNode(address: string) {
    const method = this.queryContract().methods.removeNode(address);
    return await this.sendTransaction(method, this.contractConfig.queryAddress);
  }

  async getQueryNode(address: string): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.queryContract().methods.getNode(address).call();
  }

  async queryNodeList(): Promise<string[]> {
    return this.queryContract().methods.nodeList().call();
  }

  async getQueryAccount(
    user: string,
    node: string
  ): Promise<{
    user: string;
    node: string;
    nonce: bigint;
    balance: bigint;
    pendingRefund: bigint;
    refunds: {
      index: bigint;
      amount: bigint;
      createdAt: bigint;
      processed: boolean;
    }[];
  }> {
    return this.queryContract().methods.getAccount(user, node).call();
  }

  async querySettlementFees(data: SettlementData) {
    const messageHash = Web3.utils.keccak256(data.abiEncode());
    const signature = this.web3.eth.accounts.sign(
      messageHash,
      this.account.privateKey
    );

    const settlement = {
      signature: signature.signature,
      data: {
        id: data.id,
        user: data.user,
        cost: data.cost,
        nonce: data.nonce,
        userSignature: data.userSignature,
      },
    };

    const method = this.queryContract().methods.settlementFees(settlement);
    return await this.sendTransaction(method, this.contractConfig.queryAddress);
  }

  async addInferenceNode(address: string, url: string, public_key: string) {
    const method = this.inferenceContract().methods.addNode(
      address,
      url,
      public_key
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.inferenceAddress
    );
  }

  async removeInferenceNode(address: string) {
    const method = this.inferenceContract().methods.removeNode(address);
    return await this.sendTransaction(
      method,
      this.contractConfig.inferenceAddress
    );
  }

  async getInferenceNode(address: string): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.inferenceContract().methods.getNode(address).call();
  }

  async inferenceNodeList(): Promise<string[]> {
    return this.inferenceContract().methods.nodeList().call();
  }

  async getInferenceAccount(
    user: string,
    node: string
  ): Promise<{
    user: string;
    node: string;
    nonce: bigint;
    balance: bigint;
    pendingRefund: bigint;
    refunds: {
      index: bigint;
      amount: bigint;
      createdAt: bigint;
      processed: boolean;
    }[];
  }> {
    return this.inferenceContract().methods.getAccount(user, node).call();
  }

  async inferenceSettlementFees(data: SettlementData) {
    const messageHash = Web3.utils.keccak256(data.abiEncode());
    const signature = this.web3.eth.accounts.sign(
      messageHash,
      this.account.privateKey
    );

    const settlement = {
      signature: signature.signature,
      data: {
        id: data.id,
        user: data.user,
        cost: data.cost,
        nonce: data.nonce,
        userSignature: data.userSignature,
      },
    };

    const method = this.inferenceContract().methods.settlementFees(settlement);
    return await this.sendTransaction(
      method,
      this.contractConfig.inferenceAddress
    );
  }

  async addTrainingNode(address: string, url: string, public_key: string) {
    const method = this.trainingContract().methods.addNode(
      address,
      url,
      public_key
    );
    return await this.sendTransaction(
      method,
      this.contractConfig.trainingAddress
    );
  }

  async removeTrainingNode(address: string) {
    const method = this.trainingContract().methods.removeNode(address);
    return await this.sendTransaction(
      method,
      this.contractConfig.trainingAddress
    );
  }

  async getTrainingNode(address: string): Promise<{
    nodeAddress: string;
    url: string;
    status: number;
    amount: string;
    jobsCount: bigint;
    publicKey: string;
  }> {
    return this.trainingContract().methods.getNode(address).call();
  }

  async trainingNodeList(): Promise<string[]> {
    return this.trainingContract().methods.nodeList().call();
  }

  async getTrainingAccount(
    user: string,
    node: string
  ): Promise<{
    user: string;
    node: string;
    nonce: bigint;
    balance: bigint;
    pendingRefund: bigint;
    refunds: {
      index: bigint;
      amount: bigint;
      createdAt: bigint;
      processed: boolean;
    }[];
  }> {
    return this.trainingContract().methods.getAccount(user, node).call();
  }

  async trainingSettlementFees(data: SettlementData) {
    const messageHash = Web3.utils.keccak256(data.abiEncode());
    const signature = this.web3.eth.accounts.sign(
      messageHash,
      this.account.privateKey
    );

    const settlement = {
      signature: signature.signature,
      data: {
        id: data.id,
        user: data.user,
        cost: data.cost,
        nonce: data.nonce,
        userSignature: data.userSignature,
      },
    };

    const method = this.trainingContract().methods.settlementFees(settlement);
    return await this.sendTransaction(
      method,
      this.contractConfig.trainingAddress
    );
  }

  public async getRequestHeaders(
    node: string,
    fileId?: BigInt,
    nonce?: BigInt
  ): Promise<Record<string, string>> {
    const generatedNonce = nonce ?? this.secureNonce();
    const request = new SettlementRequest(
      generatedNonce,
      this.getWallet().address,
      node,
      fileId
    );
    const signature = request.generateSignature(this.getWallet().privateKey);
    return signature.toRequestHeaders();
  }

  private secureNonce(): BigInt {
    const timestampMs = Date.now();
    const randomBytesBuffer = randomBytes(4);
    const randomInt = randomBytesBuffer.readUInt32BE(0);
    const randomPart = randomInt % 100000;
    return BigInt(timestampMs) * BigInt(100000) + BigInt(randomPart);
  }
}
