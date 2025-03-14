import type { Action, Content, ModelProviderName, Plugin } from "@elizaos/core";
import type { Tool } from "alith";
import { ElizaOSAgentRuntime } from "./agent";

export class HandlerError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "HandlerError";
	}
}

export function convertActionToTool(
	provider: ModelProviderName,
	action: Action,
): Tool {
	return {
		name: action.name,
		description: action.description,
		parameters: {
			type: "object",
		},
		handler: async () => {
			const runtime = await ElizaOSAgentRuntime.set(provider);
			const [memory, state] = ElizaOSAgentRuntime.getMemoryAndState(provider);
			let message = null as Content | null;
			if (!(await action.validate(runtime, memory, state))) {
				throw new HandlerError("handler validate error");
			}
			if (
				!(await action.handler(
					runtime,
					memory,
					state,
					{},
					async (newMessages) => {
						message = newMessages;
						return [memory];
					},
				))
			) {
				throw new HandlerError("handler run failed");
			}
			return message?.text ?? "";
		},
	};
}

export function convertPluginsToTools(
	provider: ModelProviderName,
	plugins: Plugin[],
): Tool[] {
	const tools = [];
	for (const plugin of plugins ?? []) {
		for (const tool of convertPluginToTools(provider, plugin)) {
			tools.push(tool);
		}
	}
	return tools;
}

export function convertPluginToTools(
	provider: ModelProviderName,
	plugin: Plugin,
): Tool[] {
	const tools = [];
	for (const action of plugin.actions ?? []) {
		tools.push(convertActionToTool(provider, action));
	}
	return tools;
}
