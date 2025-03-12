/**
 * Main exports for the CDP Alith package
 */

import { z } from "zod";
import { AgentKit, type Action } from "@coinbase/agentkit";
import { Tool } from "alith";

/**
 * Get Alith tools from an AgentKit instance
 *
 * @param agentKit - The AgentKit instance
 * @returns An array of Alith tools
 */
export function getAlithAITools(agentKit: AgentKit): Tool[] {
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

function parseArgs<TActionSchema extends z.ZodTypeAny = z.ZodTypeAny>(
	argsSchema: TActionSchema,
	...args: any[]
): z.infer<TActionSchema> {
	// If the schema is not an object, parse the arguments directly
	if (!(argsSchema instanceof z.ZodObject)) {
		return argsSchema.parse(args[0]); // Assume the first argument is the value
	}

	// If the schema is an object, parse the arguments into an object
	const properties = argsSchema.shape;
	const argsObject: Record<string, any> = {};
	let index = 0;
	for (const key in properties) {
		if (properties.hasOwnProperty(key)) {
			const value =
				args[index] !== undefined
					? args[index]
					: properties[key] instanceof z.ZodString
						? ""
						: properties[key] instanceof z.ZodNumber
							? 0
							: properties[key] instanceof z.ZodBoolean
								? false
								: null;
			argsObject[key] = value;
			index++;
		}
	}
	return argsSchema.parse(argsObject);
}
