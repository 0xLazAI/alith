import type {
	Action,
	ActionExample,
	Handler,
	HandlerCallback,
	IAgentRuntime,
	Memory,
	State,
	Validator,
} from "@elizaos/core";
import type { Agent } from "alith";

const promptActionExamples = [
	[
		{
			user: "{{user1}}",
			content: {
				text: "Prompt with the alith agent: 'You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question. Then Calculate 10 - 3'",
			},
		},
		{
			user: "{{agentName}}",
			content: {
				text: "7",
				action: "alith_prompt_action",
			},
		},
	],
];

export class AlithAction implements Action {
	agent: Agent;
	name: string;
	description: string;
	examples: ActionExample[][];
	similes: string[];
	handler: Handler;
	validate: Validator;
	suppressInitialMessage?: boolean;

	constructor(agent: Agent) {
		this.agent = agent;
		this.name = "alith_prompt_action";
		this.description =
			"Call the alith action to prompt with the alith agent with tools.";
		this.examples = promptActionExamples;
		this.similes = [
			"alith_prompt_action",
			"alith_prompt",
			"alith",
			"alith_agent",
			"alith_agent_prompt",
		];
		this.validate = async (_runtime: IAgentRuntime, _message: Memory) => {
			return true;
		};
		this.handler = async (
			_runtime: IAgentRuntime,
			message: Memory,
			_state?: State,
			_options?: { [key: string]: unknown },
			callback?: HandlerCallback,
		): Promise<boolean> => {
			const text = this.agent.prompt(message.content.text);
			if (callback) {
				callback({
					text,
				});
			}
			return true;
		};
	}
}
