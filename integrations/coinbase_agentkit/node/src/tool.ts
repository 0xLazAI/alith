/**
 * Main exports for the CDP Alith package
 */

import { z } from "zod";
import { AgentKit, type Action } from "@coinbase/agentkit";
import { Tool, parseArgs } from "alith";

/**
 * Get Alith tools from an AgentKit instance
 *
 * @param agentKit - The AgentKit instance
 * @returns An array of Alith tools
 */
export function getAlithTools(agentKit: AgentKit): Tool[] {
	const actions: Action[] = agentKit.getActions();
	const tools: Tool[] = [];
	for (const action of actions ?? []) {
		tools.push(convertActionToTool(action));
	}
	return tools;
}

function convertActionToTool(action: Action): Tool {
	return {
		name: action.name,
		description: action.description,
		parameters: action.schema,
		handler: async (...args: any[]) => {
			const actionArgs = parseArgs(action.schema, ...args);
			return await action.invoke(actionArgs);
		},
	};
}
