import * as openpgp from "openpgp";

export async function encrypt(
  data: Uint8Array,
  password: string
): Promise<Uint8Array> {
  const message = await openpgp.createMessage({
    binary: data,
  });

  const encrypted = await openpgp.encrypt({
    message,
    passwords: [password],
    format: "binary",
  });

  return new Uint8Array(encrypted);
}

export async function decrypt(
  data: Uint8Array,
  password: string
): Promise<Uint8Array> {
  const message = await openpgp.readMessage({
    binaryMessage: data,
  });

  const decrypted = await openpgp.decrypt({
    message,
    passwords: [password],
    format: "binary",
  });

  return new Uint8Array(decrypted.data);
}
