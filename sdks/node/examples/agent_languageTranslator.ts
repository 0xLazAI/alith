import { Agent } from "alith";

const frenchTranslatorAgent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "Your-api-key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: `You are a professional English-to-French translator. 
  You accurately translate English text into natural, fluent, and grammatically correct French. 
  Preserve the tone, emotion, and context of the original text. 
  Do not explain or add extra commentary—return only the translated text.`
});

// Example user input
const translationRequest = {
  text: "Artificial intelligence is transforming the way we live and work.",
  tone: "formal" // can be "formal", "casual", "neutral", etc.
};

async function translateToFrench() {
  const response = await frenchTranslatorAgent.prompt(`
    Translate the following English text into French.
    Maintain a ${translationRequest.tone || "neutral"} tone and ensure the translation sounds natural to native French speakers.

    Text to translate:
    "${translationRequest.text}"
  `);

  console.log("🇫🇷 Translated Text:\n", response);
}

translateToFrench();
