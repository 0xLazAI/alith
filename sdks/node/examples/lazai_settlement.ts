import { Client, ChainConfig } from 'alith/lazai'

const node = '0x1122330000000000000000000000000000000000'
const client = new Client(ChainConfig.local())
await client.addInferenceNode(node, 'url', 'node pub key')
await client.addUser(100000)
await client.deposit(1000000)
await client.depositInference(node, 10)
console.log('The inference account of user is', await client.getInferenceAccount(client.getWallet().address, node))
