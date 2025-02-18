import type { Plugin } from "@elizaos/core";
import { AlithAction } from "./actions";
import { Agent } from "alith";

function createAlithPlugin(agent: Agent): Plugin {
	return {
		name: "alith",
		description: "Plugin for interacting with the alith agent.",
		actions: [new AlithAction(agent)],
	};
}

export { createAlithPlugin };
