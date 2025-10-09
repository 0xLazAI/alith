import { Agent } from "alith";

const articleAgent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "Your-api-key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: "You are an expert article writer who produces well-structured, engaging, and SEO-friendly articles. Always provide clear section titles, concise paragraphs, and maintain a professional tone."
});

// Example user input
const articleRequest = {
  topic: "The Future of Artificial Intelligence",
  targetAudience: "tech enthusiasts",
  tone: "informative and futuristic",
  wordCount: 600
};

async function generateArticle() {
  const response = await articleAgent.prompt(`
    Topic: ${articleRequest.topic}
    Target Audience: ${articleRequest.targetAudience}
    Tone: ${articleRequest.tone}
    Desired Word Count: ${articleRequest.wordCount}

    Write a complete article following these rules:
    1. Include a title and 4–6 clear sections with headings.
    2. Use engaging and professional language.
    3. Maintain the requested tone.
    4. Keep total length around ${articleRequest.wordCount} words.
    5. End with a short conclusion or takeaway.
  `);

  console.log(response);
}

generateArticle();