import { ChainManager, ChainConfig } from './chain'
import { ProofData, SettlementProofData } from './proof'
import {
  AI_PROCESS_CONTRACT_ABI,
  ContractConfig,
  DATA_REGISTRY_CONTRACT_ABI,
  VERIFIED_COMPUTING_CONTRACT_ABI,
} from './contracts'

import Web3 from 'web3'

export class Client extends ChainManager {
  contractConfig: ContractConfig

  constructor(
    chainConfig: ChainConfig = ChainConfig.testnet(),
    contractConfig: ContractConfig = new ContractConfig(),
    privateKey: string = process.env.PRIVATE_KEY || '',
  ) {
    super(chainConfig, privateKey)
    this.contractConfig = contractConfig
  }

  dataRegistryContract() {
    return new this.web3.eth.Contract(DATA_REGISTRY_CONTRACT_ABI, this.contractConfig.dataRegistryAddress)
  }

  verifiedComputingContract() {
    return new this.web3.eth.Contract(VERIFIED_COMPUTING_CONTRACT_ABI, this.contractConfig.verifiedComputingAddress)
  }

  dataAnchorTokenContract() {
    return new this.web3.eth.Contract(VERIFIED_COMPUTING_CONTRACT_ABI, this.contractConfig.dataAnchorTokenAddress)
  }

  inferenceContract() {
    return new this.web3.eth.Contract(AI_PROCESS_CONTRACT_ABI, this.contractConfig.inferenceAddress)
  }

  trainingContract() {
    return new this.web3.eth.Contract(AI_PROCESS_CONTRACT_ABI, this.contractConfig.trainingAddress)
  }

  settlementContract() {
    return new this.web3.eth.Contract(AI_PROCESS_CONTRACT_ABI, this.contractConfig.settlementAddress)
  }

  getWallet() {
    return this.account
  }

  async addFile(url: string): Promise<BigInt> {
    const method = this.dataRegistryContract().methods.addFile(url)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
    return this.getFileIdByUrl(url)
  }

  async getFileIdByUrl(url: string): Promise<BigInt> {
    return this.dataRegistryContract().methods.getFileIdByUrl(url).call()
  }

  async addNode(address: string, url: string, publicKey: string) {
    const method = this.verifiedComputingContract().methods.addNode(address, url, publicKey)
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async addProof(fileId: BigInt, data: ProofData) {
    const packedData = data.abiEncode()
    const messageHash = Web3.utils.keccak256(packedData)
    const signature = this.web3.eth.accounts.sign(messageHash, this.account.privateKey)

    const proof = {
      signature: signature.signature,
      data: {
        id: data.id,
        fileUrl: data.fileUrl,
        proofUrl: data.proofUrl,
      },
    }

    const method = this.dataRegistryContract().methods.addProof(fileId, proof)
    return await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async addFileWithPermissions(
    url: string,
    ownerAddress: string,
    permissions: { account: string; key: string }[],
  ): Promise<BigInt> {
    const method = this.dataRegistryContract().methods.addFileWithPermissions(url, ownerAddress, permissions)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
    return this.getFileIdByUrl(url)
  }

  async addPermissionForFile(fileId: BigInt, account: string, key: string) {
    const method = this.dataRegistryContract().methods.addPermissionForFile(fileId, account, key)
    return await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async getFile(fileId: BigInt): Promise<{
    id: BigInt
    url: string
    owner: string
    createdAt: BigInt
  }> {
    return this.dataRegistryContract().methods.getFile(fileId).call()
  }

  async getFilePermission(fileId: BigInt, account: string): Promise<string> {
    return this.dataRegistryContract().methods.getFilePermission(fileId, account).call()
  }

  async getFileProof(
    fileId: BigInt,
    index: BigInt,
  ): Promise<{
    timestamp: BigInt
    hash: string
    signature: string
  }> {
    return this.dataRegistryContract().methods.getFileProof(fileId, index).call()
  }

  async filesCount(): Promise<BigInt> {
    return this.dataRegistryContract().methods.filesCount().call()
  }

  async requestReward(fileId: BigInt, proofIndex: BigInt = BigInt(1)) {
    const method = this.dataRegistryContract().methods.requestReward(fileId, proofIndex)
    return await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async removeNode(address: string) {
    const method = this.verifiedComputingContract().methods.removeNode(address)
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async nodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.nodeList().call()
  }

  async getNode(address: string): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: BigInt
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.getNode(address).call()
  }

  async updateNodeFee(fee: BigInt) {
    const method = this.verifiedComputingContract().methods.updateNodeFee(fee)
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async nodeFee(): Promise<BigInt> {
    return this.verifiedComputingContract().methods.nodeFee().call()
  }

  async requestProof(fileId: BigInt, value: BigInt = BigInt(0)) {
    const method = this.verifiedComputingContract().methods.requestProof(fileId)
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress, value.toString())
  }

  async completeJob(jobId: BigInt) {
    const method = this.verifiedComputingContract().methods.completeJob(jobId)
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async getJob(jobId: BigInt): Promise<{
    fileId: BigInt
    bidAmount: string
    status: number
    addedTimestamp: BigInt
    ownerAddress: string
    nodeAddress: string
  }> {
    return this.verifiedComputingContract().methods.getJob(jobId).call()
  }

  async fileJobIds(fileId: BigInt): Promise<BigInt[]> {
    return this.verifiedComputingContract().methods.fileJobIds(fileId).call()
  }

  async jobsCount(): Promise<BigInt> {
    return this.verifiedComputingContract().methods.jobsCount().call()
  }

  async nodeListAt(index: BigInt): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: BigInt
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.nodeListAt(index).call()
  }

  async activeNodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.activeNodeList().call()
  }

  async activeNodeListAt(index: BigInt): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: BigInt
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.activeNodeListAt(index).call()
  }

  async nodesCount(): Promise<BigInt> {
    return this.verifiedComputingContract().methods.nodesCount().call()
  }

  async activeNodesCount(): Promise<BigInt> {
    return this.verifiedComputingContract().methods.activeNodesCount().call()
  }

  async isNode(address: string): Promise<boolean> {
    return this.verifiedComputingContract().methods.isNode(address).call()
  }

  async submitJob(fileId: BigInt, value: BigInt): Promise<void> {
    const method = this.verifiedComputingContract().methods.submitJob(fileId)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress, value.toString())
  }

  async claim() {
    const method = this.verifiedComputingContract().methods.claim()
    return await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  /**
   * Mint a new Data Anchor Token (DAT) with the specified parameters.
   */
  async mintDAT() {
    const method = this.dataAnchorTokenContract().methods.mint
    return await this.sendTransaction(method, this.contractConfig.dataAnchorTokenAddress)
  }

  /**
   * Returns the balance of a specific Data Anchor Token (DAT) for a given account and token ID.
   */
  async getDATBalance(account: string, id: BigInt): Promise<BigInt> {
    return this.dataAnchorTokenContract().methods.balanceOf(account, id).call()
  }

  /**
   * Returns the Uri for a specific Data Anchor Token (DAT) by its token ID.
   */
  async dataUri(tokenId: BigInt): Promise<string> {
    return this.dataAnchorTokenContract().methods.uri(tokenId).call()
  }

  async getUser(address: string): Promise<{
    addr: string
    availableBalance: BigInt
    totalBalance: BigInt
    inferenceNodes: string[]
    trainingNodes: string[]
  }> {
    return this.settlementContract().methods.getUser(address).call()
  }

  async getAllUsers(): Promise<
    {
      addr: string
      availableBalance: BigInt
      totalBalance: BigInt
      inferenceNodes: string[]
      trainingNodes: string[]
    }[]
  > {
    return this.settlementContract().methods.getAllUser().call()
  }

  async addUser(amount: number | string) {
    const method = this.settlementContract().methods.addUser()
    return await this.sendTransaction(method, this.contractConfig.settlementAddress, amount)
  }

  async deleteUser() {
    const method = this.settlementContract().methods.deleteUser()
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async deposit(amount: number | string) {
    const method = this.settlementContract().methods.deposit()
    return await this.sendTransaction(method, this.contractConfig.settlementAddress, amount)
  }

  async withdraw(amount: number | string) {
    const method = this.settlementContract().methods.withdraw(amount)
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async depositTraining(node: string, amount: number) {
    const method = this.settlementContract().methods.depositTraining(node, amount)
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async depositInference(node: string, amount: number) {
    const method = this.settlementContract().methods.depositInference(node, amount)
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async retrieveTraining(nodes: string[]) {
    const method = this.settlementContract().methods.retrieveTraining(nodes)
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async retrieveInference(nodes: string[]) {
    const method = this.settlementContract().methods.retrieveInference(nodes)
    return await this.sendTransaction(method, this.contractConfig.settlementAddress)
  }

  async addInferenceNode(address: string, url: string, public_key: string) {
    const method = this.inferenceContract().methods.addNode(address, url, public_key)
    return await this.sendTransaction(method, this.contractConfig.inferenceAddress)
  }

  async removeInferenceNode(address: string) {
    const method = this.inferenceContract().methods.removeNode(address)
    return await this.sendTransaction(method, this.contractConfig.inferenceAddress)
  }

  async getInferenceNode(address: string): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: BigInt
    publicKey: string
  }> {
    return this.inferenceContract().methods.getNode(address).call()
  }

  async inferenceNodeList(): Promise<string[]> {
    return this.inferenceContract().methods.nodeList().call()
  }

  async getInferenceAccount(
    user: string,
    node: string,
  ): Promise<{
    user: string
    node: string
    nonce: BigInt
    balance: BigInt
    pendingRefund: BigInt
    refunds: {
      index: BigInt
      amount: BigInt
      createdAt: BigInt
      processed: boolean
    }[]
  }> {
    return this.inferenceContract().methods.getAccount(user, node).call()
  }

  async inferenceSettlementFees(user: string, cost: number) {
    const data = new SettlementProofData(0, user, cost, 0)
    const packedData = data.abiEncode()
    const messageHash = Web3.utils.keccak256(packedData)
    const signature = this.web3.eth.accounts.sign(messageHash, this.account.privateKey)

    const proof = {
      signature: signature.signature,
      data: {
        id: data.id,
        user: data.user,
        cost: data.cost,
        nonce: data.nonce,
      },
    }

    const method = this.inferenceContract().methods.settlementFees(proof)
    return await this.sendTransaction(method, this.contractConfig.inferenceAddress)
  }

  async addTrainingNode(address: string, url: string, public_key: string) {
    const method = this.trainingContract().methods.addNode(address, url, public_key)
    return await this.sendTransaction(method, this.contractConfig.trainingAddress)
  }

  async removeTrainingNode(address: string) {
    const method = this.trainingContract().methods.removeNode(address)
    return await this.sendTransaction(method, this.contractConfig.trainingAddress)
  }

  async getTrainingNode(address: string): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: BigInt
    publicKey: string
  }> {
    return this.trainingContract().methods.getNode(address).call()
  }

  async trainingNodeList(): Promise<string[]> {
    return this.trainingContract().methods.nodeList().call()
  }

  async getTrainingAccount(
    user: string,
    node: string,
  ): Promise<{
    user: string
    node: string
    nonce: BigInt
    balance: BigInt
    pendingRefund: BigInt
    refunds: {
      index: BigInt
      amount: BigInt
      createdAt: BigInt
      processed: boolean
    }[]
  }> {
    return this.trainingContract().methods.getAccount(user, node).call()
  }

  async trainingSettlementFees(user: string, cost: number) {
    const data = new SettlementProofData(0, user, cost, 0)
    const packedData = data.abiEncode()
    const messageHash = Web3.utils.keccak256(packedData)
    const signature = this.web3.eth.accounts.sign(messageHash, this.account.privateKey)

    const proof = {
      signature: signature.signature,
      data: {
        id: data.id,
        user: data.user,
        cost: data.cost,
        nonce: data.nonce,
      },
    }

    const method = this.trainingContract().methods.settlementFees(proof)
    return await this.sendTransaction(method, this.contractConfig.trainingAddress)
  }
}
