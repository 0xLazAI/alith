import { Agent, Extractor } from "alith";
import { z } from "zod";

export const personSchema = z
  .object({
    name: z.string(),
    age: z.number(),
  })
  .strip();

const agent = new Agent({ model: "gpt-4" });
const extractor = new Extractor(agent, personSchema);
console.log(extractor.extract("Alice is 18 years old!"));
