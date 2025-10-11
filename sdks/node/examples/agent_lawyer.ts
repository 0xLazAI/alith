import { Agent } from "alith";

const lawyerAgent = new Agent({
  model: "llama-3.1-8b-instant",
  apiKey: "Your-api-key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: `You are an experienced legal expert and lawyer.
  Your job is to analyze legal problems and provide:
  - The relevant legal sections, acts, or laws that apply (based on Indian law, unless otherwise specified).
  - The rights the person has in that context.
  - A clear and concise explanation of possible legal actions or remedies.
  
  Be professional, accurate, and easy to understand.
  Use simple terms when possible, but cite correct legal sections.
  Do not give irrelevant commentary or moral advice — only legal guidance.`
});

// Example user input (you can replace this dynamically)
const userCase = {
  issue: "My landlord is refusing to return my security deposit even though I moved out and left the house in good condition."
};

async function consultLawyer() {
  const response = await lawyerAgent.prompt(`
    A person is facing the following legal issue:
    "${userCase.issue}"

    Identify:
    1. The relevant legal sections or acts that apply.
    2. The person's rights in this situation.
    3. The legal remedies or actions they can take.
  `);

  console.log("⚖️ Legal Advice:\n", response);
}

consultLawyer();
