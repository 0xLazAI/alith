import { Client, ProofData } from "alith/lazai";

async function main() {
  const contributor = new Client();
  const node = new Client();
  console.log("wallet address", contributor.getWallet().address);
  const url = "https://example.com/okyes.txt";
  let fileId = await contributor.getFileIdByUrl(url);
  // File not found, add it
  if (fileId === BigInt(0)) {
    fileId = await contributor.addFile(url);
  }
  console.log("file id:", fileId);
  let nodeInfo = await node.getNode(node.getWallet().address);
  if (nodeInfo.url.length > 0) await node.removeNode(node.getWallet().address);
  await node.addNode(
    node.getWallet().address,
    "https://example.com/node",
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
  );
  const nodeFee = BigInt(10);
  await node.updateNodeFee(nodeFee);
  await contributor.requestProof(fileId, nodeFee);
  const ids = await contributor.fileJobIds(fileId);
  console.log("file proof job ids", ids);
  const job_id = ids[ids.length - 1];
  const job = await contributor.getJob(job_id);
  console.log("job id", job_id);
  console.log("job", job);
  nodeInfo = await contributor.getNode(job.nodeAddress);
  console.log("node info:", nodeInfo);
  await node.completeJob(job_id);
  console.log("completed job", job_id);
  await node.addProof(fileId, new ProofData(Number(fileId), "", ""));
  console.log("proof added for file", fileId);
  await contributor.requestReward(fileId);
  console.log("reward requested for file", fileId);
}
await main();
