import { ChainConfig, Client } from "alith/lazai";
import { Agent } from "alith";

const node = "0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194";
const client = new Client(ChainConfig.local());
await client.addUser(100000000);
await client.deposit(10000000);
await client.depositInference(node, 10);
console.log(
  "The inference account of user is",
  await client.getInferenceAccount(client.getWallet().address, node)
);
const fileId = 1;
const nodeInfo = await client.getInferenceNode(node);
const url = nodeInfo.url;
const headers = await client.getRequestHeaders(node, BigInt(fileId));
