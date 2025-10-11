import { Agent } from "alith";

const subjectAssistantAgent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "your-api-key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: `You are an intelligent and knowledgeable assistant. 
  You provide clear, accurate, and well-structured answers to user questions across various subjects 
  such as science, technology, history, literature, geography, and general knowledge.
  Always provide concise, factual, and easy-to-understand explanations.
  At the end of each response, include 2–3 relevant and credible reference links (Wikipedia, academic sites, or official sources).
  Do not invent or fabricate links — only use real, verifiable sources.`
});

// Example user question
const userQuestion = {
  topic: "quantum computing",
  question: "What is quantum entanglement and why is it important in quantum computing?"
};

async function getSubjectAnswer() {
  const response = await subjectAssistantAgent.prompt(`
    Subject: ${userQuestion.topic}
    Question: ${userQuestion.question}

    Provide a well-structured explanation suitable for a general audience.
    End the answer with 2–3 real reference links.
  `);

  console.log("📘 Assistant Answer:\n", response);
}

getSubjectAnswer();