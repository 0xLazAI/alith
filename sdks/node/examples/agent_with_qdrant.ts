import {
  type Embeddings,
  QdrantStore,
  RemoteModelEmbeddings,
  type Store,
} from "alith";

const embeddings: Embeddings = new RemoteModelEmbeddings(
  "your embeddings model name",
  "your API key",
  "base url"
);
const store: Store = new QdrantStore(embeddings);
store.save("Hello, World");
console.log(store.search("Hello, World"));
