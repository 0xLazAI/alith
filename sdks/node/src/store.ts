import { QdrantClient, QdrantClientParams } from "@qdrant/js-client-rest";
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

export { type Store, QdrantStore, QdrantClient, QdrantClientParams };
