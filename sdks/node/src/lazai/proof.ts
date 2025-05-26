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
