import { PineconeStore, RemoteModelEmbeddings } from "../src";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config();

async function main() {
  // Create embeddings model (using OpenAI's text-embedding-3-small)
  const embeddings = new RemoteModelEmbeddings(
    "text-embedding-3-small",
    process.env.OPENAI_API_KEY || "your-openai-api-key",
    "https://api.openai.com/v1"
  );

  // Initialize Pinecone store
  const store = new PineconeStore(
    embeddings,
    process.env.PINECONE_API_KEY || "your-pinecone-api-key",
    "alith", // index name
    "default" // namespace
  );

  console.log("📌 Pinecone Store Example\n");

  // Save a single document
  console.log("💾 Saving documents...");
  await store.save("Artificial Intelligence is transforming the world");

  // Save multiple documents
  await store.saveDocs([
    "Machine Learning is a subset of AI that focuses on learning from data",
    "Deep Learning uses neural networks with multiple layers",
    "Natural Language Processing enables AI to understand and generate human language",
    "Computer Vision allows machines to interpret visual information",
  ]);

  console.log("✅ Documents saved successfully!\n");

  // Search for similar documents
  console.log("🔍 Searching for documents similar to 'What is AI?'...\n");
  const results = await store.search("What is AI?", 3, 0.5);

  results.forEach((result, index) => {
    console.log(`${index + 1}. ${result}`);
  });

  console.log("\n✨ Search completed!");
}

main().catch(console.error);
