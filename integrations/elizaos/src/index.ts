import type { Plugin } from "@elizaos/core";
import { AlithAction } from "./actions";
import { Agent, AgentOptions } from "./agent";

function createAlithPlugin(agent: Agent): Plugin {
	return {
		name: "alith",
		description: "Plugin for interacting with the alith agent.",
		actions: [new AlithAction(agent)],
	};
}

export { createAlithPlugin, Agent, AgentOptions };
