import axios from "axios";

interface Embeddings {
  embedTexts(texts: string[]): Promise<number[][]>;
}

class RemoteModelEmbeddings implements Embeddings {
  private model: string;
  private apiKey: string;
  private baseUrl: string;
  private port?: number | string;

  constructor(
    model: string,
    apiKey: string,
    baseUrl: string,
    port?: number | string
  ) {
    this.model = model;
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.port = port;
  }

  async embedTexts(texts: string[]): Promise<number[][]> {
    let url: string;
    if (this.baseUrl.startsWith("http")) {
      url = this.port
        ? `${this.baseUrl}:${this.port}/embeddings`
        : `${this.baseUrl}/embeddings`;
    } else {
      url = `https://${this.baseUrl}/embeddings`;
    }

    const headers = {
      Authorization: `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
    };

    const payload = {
      input: texts,
      model: this.model,
    };

    try {
      const response = await axios.post(url, payload, { headers });
      if (response.status === 200) {
        const responseDatas = response.data.data || [];
        const embeddings = responseDatas.map(
          (data: any) => data.embedding || []
        );
        return embeddings;
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    } catch (error) {
      throw new Error(`Error fetching embeddings: ${error}`);
    }
  }
}

export { type Embeddings, RemoteModelEmbeddings };
