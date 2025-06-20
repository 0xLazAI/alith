import { Client, ProofData } from "../src/lazai";

const client = new Client();
console.log("Wallet address", client.getWallet().address);
console.log(
  "Balance of DAT",
  await client.getDATBalance(client.getWallet().address, 1)
);
