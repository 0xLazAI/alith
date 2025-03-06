import { QdrantClient, QdrantClientParams } from '@qdrant/js-client-rest'
import { Embeddings } from './embeddings'

interface Store {
  /**
   * Searches the storage with a query, limiting the results and applying a threshold.
   *
   * @param query - The search query.
   * @param limit - The maximum number of results to return.
   * @param scoreThreshold - The minimum score threshold for results.
   * @returns A list of results matching the query.
   */
  search(query: string, limit?: number, scoreThreshold?: number): Promise<string[]>

  /**
   * Saves a value into the storage.
   *
   * @param value - The value to save.
   */
  save(value: string): Promise<void>

  /**
   * Resets the storage by clearing all stored data.
   */
  reset(): Promise<void>
}

class QdrantStore implements Store {
  private client: QdrantClient
  private collectionName: string
  private embeddings: Embeddings

  constructor(embeddings: Embeddings, collectionName: string = 'alith', params?: QdrantClientParams) {
    this.embeddings = embeddings
    this.client = new QdrantClient(params)
    this.collectionName = collectionName
  }

  async search(query: string, limit: number = 3, scoreThreshold: number = 0.4): Promise<string[]> {
    const queryVectors = await this.embedTexts([query])
    const searchResult = await this.client.search(this.collectionName, {
      vector: queryVectors[0],
      limit,
      score_threshold: scoreThreshold,
    })
    return searchResult.map((point) => point.payload?.text as string)
  }

  async save(value: string): Promise<void> {
    const vectors = await this.embedTexts([value])
    await this.client.upsert(this.collectionName, {
      points: [
        {
          id: Math.random().toString(36).substring(7),
          vector: vectors[0],
          payload: { text: value },
        },
      ],
    })
  }

  async reset(): Promise<void> {
    await this.client.deleteCollection(this.collectionName)
    await this.client.createCollection(this.collectionName, {
      vectors: {
        size: 384,
        distance: 'Cosine',
      },
    })
  }

  private async embedTexts(text: string[]): Promise<number[][]> {
    return this.embeddings.embedTexts(text)
  }
}

export { Store, QdrantStore, QdrantClient, QdrantClientParams }
