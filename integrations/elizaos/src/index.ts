import type { Plugin } from "@elizaos/core";
import { AlithAction } from "./actions";
import { Agent, AgentOptions, ElizaOSAgentRuntime } from "./agent";
import { Agent as BaseAgent } from "alith";

function createAlithPlugin(agent: BaseAgent): Plugin {
	return {
		name: "alith",
		description: "Plugin for interacting with the alith agent.",
		actions: [new AlithAction(agent)],
	};
}

export { createAlithPlugin, Agent, AgentOptions, ElizaOSAgentRuntime };
