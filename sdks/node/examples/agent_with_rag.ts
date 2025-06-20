import { Agent, QdrantStore, RemoteModelEmbeddings, type Store } from "alith";

const store: Store = new QdrantStore(
  new RemoteModelEmbeddings(
    "your embeddings model name",
    "your API key",
    "base url"
  )
);
await store.save("Hello, World");
const agent = new Agent({
  model: "gpt-4",
  preamble:
    "You are a comedian here to entertain the user using humour and jokes.",
  store,
});
console.log(await agent.prompt("Entertain me!"));
