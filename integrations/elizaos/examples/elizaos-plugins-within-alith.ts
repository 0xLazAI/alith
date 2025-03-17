import { Agent } from "elizaos-alith";

const agent = new Agent({
	model: "gpt-4",
	runtime: YourElizaOSAgentRuntime(), // Your elizaos agent runtime,
	preamble:
		"You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.",
	plugins: [
		// Put your elizaos plugins here.
	],
});
const result = agent.prompt("Calculate 10 - 3");
console.log(result);
