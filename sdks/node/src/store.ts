import { QdrantClient, QdrantClientParams } from "@qdrant/js-client-rest";
import { Pinecone } from "@pinecone-database/pinecone";
import type { Embeddings } from "./embeddings";

function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

interface Store {
  /**
   * Searches the storage with a query, limiting the results and applying a threshold.
   *
   * @param query - The search query.
   * @param limit - The maximum number of results to return.
   * @param scoreThreshold - The minimum score threshold for results.
   * @returns A list of results matching the query.
   */
  search(
    query: string,
    limit?: number,
    scoreThreshold?: number
  ): Promise<string[]>;

  /**
   * Saves a value into the storage.
   *
   * @param value - The value to save.
   */
  save(value: string): Promise<void>;

  /**
   * Resets the storage by clearing all stored data.
   */
  reset(): Promise<void>;
}

class QdrantStore implements Store {
  private client: QdrantClient;
  private collectionName: string;
  private embeddings: Embeddings;
  private vectorSize: number;

  constructor(
    embeddings: Embeddings,
    collectionName = "alith",
    vectorSize = 384,
    params?: QdrantClientParams
  ) {
    this.embeddings = embeddings;
    this.client = new QdrantClient(params);
    this.collectionName = collectionName;
    this.vectorSize = vectorSize;

    this.ensureCollectionExists().then(() => {});
  }

  private async ensureCollectionExists(): Promise<void> {
    try {
      const collections = await this.client.getCollections();
      const exists = collections.collections.some(
        (c) => c.name === this.collectionName
      );

      if (!exists) {
        await this.client.createCollection(this.collectionName, {
          vectors: {
            size: this.vectorSize,
            distance: "Cosine",
          },
        });
      }
    } catch (error) {}
  }

  async search(
    query: string,
    limit = 3,
    scoreThreshold = 0.4
  ): Promise<string[]> {
    await this.ensureCollectionExists();
    const queryVectors = await this.embedTexts([query]);
    const searchResult = await this.client.search(this.collectionName, {
      vector: queryVectors[0],
      limit,
      score_threshold: scoreThreshold,
    });
    return searchResult.map((point) => point.payload?.text as string);
  }

  async save(value: string): Promise<void> {
    await this.ensureCollectionExists();
    const vectors = await this.embedTexts([value]);
    await this.client.upsert(this.collectionName, {
      points: [
        {
          id: generateUUID(),
          vector: vectors[0],
          payload: { text: value },
        },
      ],
    });
  }

  async reset(): Promise<void> {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some(
      (c) => c.name === this.collectionName
    );
    if (exists) {
      await this.client.deleteCollection(this.collectionName);
    }
    await this.client.createCollection(this.collectionName, {
      vectors: {
        size: this.vectorSize,
        distance: "Cosine",
      },
    });
  }

  async saveDocs(values: string[]): Promise<void> {
    await this.ensureCollectionExists();
    const vectors = await this.embedTexts(values);
    const points = values.map((value, index) => ({
      id: generateUUID(),
      vector: vectors[index],
      payload: { text: value },
    }));
    await this.client.upsert(this.collectionName, {
      points,
    });
  }

  private async embedTexts(text: string[]): Promise<number[][]> {
    return this.embeddings.embedTexts(text);
  }
}

class PineconeStore implements Store {
  private client: Pinecone;
  private indexName: string;
  private embeddings: Embeddings;
  private namespace: string;

  constructor(
    embeddings: Embeddings,
    apiKey: string,
    indexName = "alith",
    namespace = "default"
  ) {
    this.embeddings = embeddings;
    this.client = new Pinecone({ apiKey });
    this.indexName = indexName;
    this.namespace = namespace;
  }

  async search(
    query: string,
    limit = 3,
    scoreThreshold = 0.4
  ): Promise<string[]> {
    const index = this.client.index(this.indexName);
    const queryVectors = await this.embedTexts([query]);

    const queryResponse = await index.namespace(this.namespace).query({
      vector: queryVectors[0],
      topK: limit,
      includeMetadata: true,
    });

    return (
      queryResponse.matches
        ?.filter((match: any) => (match.score || 0) >= scoreThreshold)
        .map((match: any) => match.metadata?.text as string) || []
    );
  }

  async save(value: string): Promise<void> {
    const index = this.client.index(this.indexName);
    const vectors = await this.embedTexts([value]);
    const id = generateUUID();

    await index.namespace(this.namespace).upsert([
      {
        id,
        values: vectors[0],
        metadata: { text: value },
      },
    ]);
  }

  async reset(): Promise<void> {
    const index = this.client.index(this.indexName);
    await index.namespace(this.namespace).deleteAll();
  }

  async saveDocs(values: string[]): Promise<void> {
    const index = this.client.index(this.indexName);
    const vectors = await this.embedTexts(values);

    const records = values.map((value, i) => ({
      id: generateUUID(),
      values: vectors[i],
      metadata: { text: value },
    }));

    // Pinecone has a limit of 100 vectors per upsert, so batch them
    const batchSize = 100;
    for (let i = 0; i < records.length; i += batchSize) {
      const batch = records.slice(i, i + batchSize);
      await index.namespace(this.namespace).upsert(batch);
    }
  }

  private async embedTexts(text: string[]): Promise<number[][]> {
    return this.embeddings.embedTexts(text);
  }
}

export {
  type Store,
  QdrantStore,
  QdrantClient,
  QdrantClientParams,
  PineconeStore,
};
