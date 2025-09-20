import { Agent } from "alith";

const agent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "Your API Key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble:
    "You are a text summarization expert. Your task is to create concise, precise summaries of long text passages. Focus on the key points, main ideas, and essential information while maintaining clarity and coherence.",
});

(async () => {
  const longText = `
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, fundamentally reshaping how we interact with machines, process information, and solve complex problems across virtually every sector of human activity. From its humble beginnings in the 1950s with Alan Turing's groundbreaking work on machine intelligence and the famous Turing Test, AI has evolved through several distinct phases, each marked by significant breakthroughs, periods of optimism, and occasional setbacks known as "AI winters."

    The field encompasses a broad spectrum of techniques and approaches, including machine learning, deep learning, neural networks, natural language processing, computer vision, robotics, and expert systems. Machine learning, in particular, has become the driving force behind many of AI's recent successes, enabling systems to learn from data without being explicitly programmed for every scenario. Deep learning, a subset of machine learning inspired by the structure and function of the human brain, has revolutionized areas such as image recognition, speech processing, and language translation.

    In healthcare, AI is being used to analyze medical images with superhuman accuracy, assist in drug discovery processes that traditionally took decades, predict patient outcomes, and personalize treatment plans. The technology has shown remarkable promise in detecting early-stage cancers, identifying rare diseases, and even predicting potential health risks before symptoms appear. However, the integration of AI in healthcare also raises important questions about data privacy, algorithmic bias, and the need for regulatory frameworks to ensure patient safety.

    The business world has embraced AI for automation, predictive analytics, customer service through chatbots and virtual assistants, fraud detection, supply chain optimization, and personalized marketing. Companies like Google, Amazon, Microsoft, and Meta have invested billions of dollars in AI research and development, creating powerful platforms and tools that are now accessible to businesses of all sizes. This democratization of AI technology has led to innovative applications across industries, from agriculture and manufacturing to finance and entertainment.

    Despite these advances, AI development faces significant challenges and ethical considerations. Issues such as algorithmic bias, job displacement due to automation, privacy concerns related to data collection and usage, the potential for misuse in surveillance and autonomous weapons, and the concentration of AI power in the hands of a few large corporations have sparked intense debates among policymakers, technologists, and ethicists. The development of artificial general intelligence (AGI) – AI systems that match or exceed human cognitive abilities across all domains – remains a subject of both excitement and concern within the scientific community.

    Looking toward the future, AI is expected to continue its rapid evolution, with potential breakthroughs in quantum computing, neuromorphic processors, and brain-computer interfaces likely to accelerate progress even further. The integration of AI with other emerging technologies such as 5G networks, Internet of Things (IoT) devices, and augmented reality promises to create new possibilities for human-machine collaboration and enhance our ability to solve some of the world's most pressing challenges, including climate change, poverty, and disease.
  `;

  try {
    const summary = await agent.prompt(`Please provide a concise summary of the following text in 2-3 sentences, focusing on the key points and main ideas:\n\n${longText}`);
    
    console.log("=== ORIGINAL TEXT ===");
    console.log(`Length: ${longText.trim().length} characters`);
    console.log("\n=== SUMMARY ===");
    console.log(summary);
    console.log(`\nSummary Length: ${summary.length} characters`);
    console.log(`Compression Ratio: ${Math.round((summary.length / longText.trim().length) * 100)}%`);
  } catch (error) {
    console.error("Error generating summary:", error);
  }
})();