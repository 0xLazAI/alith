import { RemoteModelEmbeddings } from "alith";

console.log(
  new RemoteModelEmbeddings(
    "your embeddings model name",
    "your API key",
    "base url"
  ).embedTexts(["Hello", "World"])
);
