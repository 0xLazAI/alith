import { Agent } from "alith";

const imageDescriptionAgent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "your-api-key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: "You are an expert visual analyst specializing in detailed image descriptions. Provide comprehensive guidance for describing any type of image with focus on color analysis, composition, and vivid descriptive language. Include specific color terminology and detailed visual elements for all subjects."
});

async function describeImage(imageUrl: string) {
  const description = await imageDescriptionAgent.prompt(
    `Image URL: ${imageUrl}
    
    Provide a comprehensive guide for describing this image. Include:
    
    1. SUBJECT IDENTIFICATION: What type of image and main subjects
    2. DETAILED COLOR ANALYSIS: Specific colors used with proper color names (e.g., crimson, azure, emerald, etc.)
    3. COMPOSITION & ELEMENTS: Layout, positioning, textures, patterns
    4. VISUAL DETAILS: Lighting, shadows, depth, focus areas
    5. DESCRIPTIVE EXAMPLE: A complete, vivid sample description using rich vocabulary
    
    Focus on enhanced visual capacity with detailed color terminology and comprehensive analysis.`
  );
  
  console.log("🖼️ Image Description:", description);
}

// Example usage
const testImageUrl = "https://share.google/images/JvkVylrYm5s47d6Ie";
describeImage(testImageUrl);