import { encrypt } from "alith/data";
import { PinataIPFS } from "alith/data/storage";
import { Client } from "alith/lazai";
import axios, { type AxiosResponse } from "axios";
import NodeRSA from "node-rsa";
import * as crypto from "crypto";

function calculateSHA256(text: string): string {
  return crypto.createHash("sha256").update(text, "utf-8").digest("hex");
}

async function main() {
  const client = new Client();
  const ipfs = new PinataIPFS();
  // 1. Prepare your privacy data and encrypt it
  const dataFileName = "your_encrypted_data.txt";
  const privacyData = "Your Privacy Data";
  const privacyDataSha256 = calculateSHA256(privacyData);
  const encryptionSeed = "Sign to retrieve your encryption key";
  const password = client.getWallet().sign(encryptionSeed).signature;
  const encryptedData = await encrypt(Uint8Array.from(privacyData), password);
  // 2. Upload the privacy data to IPFS and get the shared url
  const token = process.env.IPFS_JWT || "";
  const fileMeta = await ipfs.upload({
    name: dataFileName,
    data: Buffer.from(encryptedData),
    token: token,
  });
  const url = await ipfs.getShareLink({ token: token, id: fileMeta.id });
  // 3. Upload the privacy url to LazAI
  let fileId = await client.getFileIdByUrl(url);
  if (fileId === BigInt(0)) {
    fileId = await client.addFileWithHash(url, privacyDataSha256);
  }
  console.log("File ID:", fileId);
  const pubKey = await client.getPublicKey();
  const rsa = new NodeRSA(pubKey, "pkcs1-public-pem");
  const encryptedKey = rsa.encrypt(password, "hex");
  await client.addPermissionForFile(
    fileId,
    client.contractConfig.dataRegistryAddress,
    encryptedKey
  );
}

await main();
