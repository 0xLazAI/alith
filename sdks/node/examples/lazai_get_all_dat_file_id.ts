import { Client } from "alith/lazai";

const client = new Client();
for (let i = 207; i <= 256; i++) {
  const url = await client.dataUri(BigInt(i));
  const fileId = await client.getFileIdByUrl(url);
  console.log(i, url, fileId);
}
