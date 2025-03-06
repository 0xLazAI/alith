import { Store, QdrantStore, Embeddings, RemoteModelEmbeddings } from 'alith'

let embeddings: Embeddings = new RemoteModelEmbeddings('your embeddings model name', 'your API key', 'base url')
const store: Store = new QdrantStore(embeddings)
store.save('Hello, World')
console.log(store.search('Hello, World'))
