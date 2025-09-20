import { Agent } from "alith";

const dietAgent = new Agent({
  model: "llama-3.1-8b-instant", 
  apiKey: "Your API Key",
  baseUrl: "https://api.groq.com/openai/v1",
  preamble: "You are a fitness and diet planning assistant. Give concise, specific recommendations with exact numbers only."
});

// Example user input
const user = {
  age: 25,
  weight: 70, // kg
  height: 175, // cm
  gender: "male",
  activity: "moderate",
  goal: "weight loss"
};

async function getDietPlan() {
  const response = await dietAgent.prompt(`
    User details: 
    Age: ${user.age}, 
    Weight: ${user.weight}kg, 
    Height: ${user.height}cm, 
    Gender: ${user.gender}, 
    Activity: ${user.activity}, 
    Goal: ${user.goal}.
    
    Provide ONLY:
    1. Daily calories needed (exact number)
    2. Exercise hours per day (exact number)
    3. Weekly exercise schedule (7 days with specific focus like "Monday: Leg Day")
    
    Keep it concise and specific.
  `);

  console.log(response);
}

getDietPlan();
