import { PinataIPFS } from "alith/data/storage";

async function main() {
  const ipfs = new PinataIPFS();
  const data = "Your privacy data";
  const name = "your_privacy_file.txt";
  const fileMeta = await ipfs.upload({
    name: name,
    data: Buffer.from(data, "utf-8"),
    token: process.env.IPFS_JWT || "",
  });
  console.log(`Upload file to the Pinata IPFS: ${fileMeta}`);
  console.log(
    `Share link: ${await ipfs.getShareLink({
      token: process.env.IPFS_JWT || "",
      id: fileMeta.id,
    })}`
  );
}

await main();
