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
  // 4. Request proof in the verified computing node
  await client.requestProof(fileId, BigInt(100));
  const jobIds = await client.fileJobIds(fileId);
  const jobId = jobIds[jobIds.length - 1];
  const job = await client.getJob(jobId);
  const nodeInfo = await client.getNode(job.nodeAddress);
  const nodeUrl = nodeInfo.url;
  const pubKey = nodeInfo.publicKey;
  const rsa = new NodeRSA(pubKey, "pkcs1-public-pem");
  const encryptedKey = rsa.encrypt(password, "hex");
  const proofRequest = {
    job_id: Number(jobId),
    file_id: Number(fileId),
    file_url: url,
    encryption_key: encryptedKey,
    encryption_seed: encryptionSeed,
    nonce: null,
    proof_url: null,
  };
  const response: AxiosResponse = await axios.post(
    `${nodeUrl}/proof`,
    proofRequest,
    {
      headers: { "Content-Type": "application/json" },
    }
  );

  if (response.status === 200) {
    console.log("Proof request sent successfully");
  } else {
    console.log("Failed to send proof request:", response.data);
  }
  // 5. Request DAT reward
  await client.requestReward(fileId);
  console.log("Reward requested for file id", fileId);
}

await main();
