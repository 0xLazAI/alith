import { ChainManager, ChainConfig } from './chain'
import { ProofData } from './proof'
import { ContractConfig, DATA_REGISTRY_CONTRACT_ABI, VERIFIED_COMPUTING_CONTRACT_ABI } from './contracts'

import Web3 from 'web3'

export class Client extends ChainManager {
  contractConfig: ContractConfig

  constructor(
    chainConfig: ChainConfig = ChainConfig.local(),
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

  getWallet() {
    return this.account
  }

  async addFile(url: string): Promise<number> {
    const method = this.dataRegistryContract().methods.addFile(url)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
    return this.getFileIdByUrl(url)
  }

  async getFileIdByUrl(url: string): Promise<number> {
    return this.dataRegistryContract().methods.getFileIdByUrl(url).call()
  }

  async addNode(address: string, url: string, publicKey: string): Promise<void> {
    const method = this.verifiedComputingContract().methods.addNode(address, url, publicKey)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async addProof(fileId: number, data: ProofData): Promise<void> {
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
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async addFileWithPermissions(
    url: string,
    ownerAddress: string,
    permissions: { account: string; key: string }[],
  ): Promise<number> {
    const method = this.dataRegistryContract().methods.addFileWithPermissions(url, ownerAddress, permissions)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
    return this.getFileIdByUrl(url)
  }

  async addPermissionForFile(fileId: number, account: string, key: string): Promise<void> {
    const method = this.dataRegistryContract().methods.addPermissionForFile(fileId, account, key)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async getFile(fileId: number): Promise<{
    id: number
    url: string
    owner: string
    createdAt: number
  }> {
    return this.dataRegistryContract().methods.getFile(fileId).call()
  }

  async getFilePermission(fileId: number, account: string): Promise<string> {
    return this.dataRegistryContract().methods.getFilePermission(fileId, account).call()
  }

  async getFileProof(
    fileId: number,
    index: number,
  ): Promise<{
    timestamp: number
    hash: string
    signature: string
  }> {
    return this.dataRegistryContract().methods.getFileProof(fileId, index).call()
  }

  async filesCount(): Promise<number> {
    return this.dataRegistryContract().methods.filesCount().call()
  }

  async requestReward(fileId: number, proofIndex: number = 1): Promise<void> {
    const method = this.dataRegistryContract().methods.requestReward(fileId, proofIndex)
    await this.sendTransaction(method, this.contractConfig.dataRegistryAddress)
  }

  async removeNode(address: string): Promise<void> {
    const method = this.verifiedComputingContract().methods.removeNode(address)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async nodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.nodeList().call()
  }

  async getNode(address: string): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: number
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.getNode(address).call()
  }

  async updateNodeFee(fee: number): Promise<void> {
    const method = this.verifiedComputingContract().methods.updateNodeFee(fee)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async requestProof(fileId: number, value: number = 0): Promise<void> {
    const method = this.verifiedComputingContract().methods.requestProof(fileId)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress, value)
  }

  async completeJob(jobId: number): Promise<void> {
    const method = this.verifiedComputingContract().methods.completeJob(jobId)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }

  async getJob(jobId: number): Promise<{
    fileId: number
    bidAmount: string
    status: number
    addedTimestamp: number
    ownerAddress: string
    nodeAddress: string
  }> {
    return this.verifiedComputingContract().methods.getJob(jobId).call()
  }

  async fileJobIds(fileId: number): Promise<number[]> {
    return this.verifiedComputingContract().methods.fileJobIds(fileId).call()
  }

  async jobsCount(): Promise<number> {
    return this.verifiedComputingContract().methods.jobsCount().call()
  }

  async nodeListAt(index: number): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: number
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.nodeListAt(index).call()
  }

  async activeNodeList(): Promise<string[]> {
    return this.verifiedComputingContract().methods.activeNodeList().call()
  }

  async activeNodeListAt(index: number): Promise<{
    nodeAddress: string
    url: string
    status: number
    amount: string
    jobsCount: number
    publicKey: string
  }> {
    return this.verifiedComputingContract().methods.activeNodeListAt(index).call()
  }

  async nodesCount(): Promise<number> {
    return this.verifiedComputingContract().methods.nodesCount().call()
  }

  async activeNodesCount(): Promise<number> {
    return this.verifiedComputingContract().methods.activeNodesCount().call()
  }

  async isNode(address: string): Promise<boolean> {
    return this.verifiedComputingContract().methods.isNode(address).call()
  }

  async submitJob(fileId: number, value: number): Promise<void> {
    const method = this.verifiedComputingContract().methods.submitJob(fileId)
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress, value)
  }

  async claim(): Promise<void> {
    const method = this.verifiedComputingContract().methods.claim()
    await this.sendTransaction(method, this.contractConfig.verifiedComputingAddress)
  }
}
