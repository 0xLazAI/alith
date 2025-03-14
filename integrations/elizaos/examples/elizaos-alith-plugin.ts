import { createAlithPlugin } from "elizaos-alith";
import { Agent } from "alith";
import { AgentRuntime } from "@elizaos/core";

const agent = new Agent({
	model: "gpt-4",
	preamble:
		"You are a comedian here to entertain the user using humour and jokes.",
});
const runtime = new AgentRuntime({
	// Add Alith plugin in the ElizaOS agent runtime.
	plugins: [
		createAlithPlugin(agent),
		// Omit other plugins
	],
	// Omit other fields
});
