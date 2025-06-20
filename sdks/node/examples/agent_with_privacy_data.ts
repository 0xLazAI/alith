import { decrypt, encrypt } from "alith/data";
import NodeRSA from "node-rsa";

async function main() {
  const privacyData = "Hello, Privacy Data with PGP!";
  const encoder = new TextEncoder();
  const dataUint8Array = encoder.encode(privacyData);
  const password = "securepassword123456789";
  const rsa = new NodeRSA({ b: 3072 });
  const encryptedKey = rsa.encrypt(password);
  console.log("Encrypted Key:", encryptedKey.toString("base64"));
  const encryptedData = await encrypt(dataUint8Array, password);
  console.log("Encrypted Data:", encryptedData.toString());
  const decryptedPassword = rsa.decrypt(encryptedKey);
  console.log("Decrypted Password:", decryptedPassword.toString());
  const decryptedData = await decrypt(
    encryptedData,
    decryptedPassword.toString()
  );
  console.log("Decrypted Data:", decryptedData.toString());
}

await main();
