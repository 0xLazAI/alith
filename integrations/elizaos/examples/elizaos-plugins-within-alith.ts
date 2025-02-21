import { Agent } from "elizaos-alith";
import { ModelProviderName } from "@elizaos/core";

const provider = ModelProviderName.OPENAI;
const agent = new Agent({
	name: "A dummy Agent",
	model: "gpt-4",
	preamble:
		"You are a comedian here to entertain the user using humour and jokes.",
	provider,
	plugins: [
		// Set your ElizaOS here.
	]
});
const result = agent.prompt("Ask a question");
