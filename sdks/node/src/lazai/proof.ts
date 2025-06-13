import { Web3 } from 'web3'

export class ProofData {
  constructor(public id: number, public fileUrl: string, public proofUrl: string) {}

  abiEncode(): string {
    const web3 = new Web3()
    return web3.eth.abi.encodeParameter(
      {
        components: [
          { name: 'id', type: 'uint256' },
          { name: 'fileUrl', type: 'string' },
          { name: 'proofUrl', type: 'string' },
        ],
        name: 'data',
        type: 'tuple',
      },
      {
        id: this.id,
        fileUrl: this.fileUrl,
        proofUrl: this.proofUrl,
      },
    )
  }
}

export class SettlementProofData {
  constructor(
    public id: string,
    public user: string,
    public cost: number,
    public nonce: number,
    public userSignature: string,
  ) {}

  abiEncode(): string {
    const web3 = new Web3()
    return web3.eth.abi.encodeParameter(
      {
        components: [
          { name: 'id', type: 'string' },
          { name: 'user', type: 'adress' },
          { name: 'cost', type: 'address' },
          { name: 'nonce', type: 'uint256' },
          { name: 'userSignature', type: 'bytes' },
        ],
        name: 'data',
        type: 'tuple',
      },
      {
        id: this.id,
        user: this.user,
        cost: this.cost,
        nonce: this.nonce,
        userSignature: this.userSignature,
      },
    )
  }
}
