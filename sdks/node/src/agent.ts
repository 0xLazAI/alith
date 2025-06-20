import { DelegateAgent, type DelegateTool } from "./internal";
import type { Memory } from "./memory";
import type { Store } from "./store";
import { TEEClient, type TEEConfig, type TEEExecutionResult } from "./tee";
import { type Tool, convertParametersToJson } from "./tool";

// Define the configuration structure for an Agent
type AgentOptions = {
	name?: string; // Optional agent name.
	model: string; // The model used by the agent.
	preamble?: string; // Introductory text or context for the agent.
	baseUrl?: string; // Optional base URL for API requests.
	apiKey?: string; // Optional API key for authentication.
	tools?: Array<Tool>; // Optional list of tools available to the agent.
	mcpConfigPath?: string; // Optional mcp config path.
	store?: Store; // Optional store for the knowledge index.
	memory?: Memory; // Optional memory for the agent conversation context.
	teeConfig?: TEEConfig; // Optional TEE configuration for secure execution.
};

// Represents an agent that can process prompts using tools
class Agent {
	private _agent: DelegateAgent;
	private _opts: AgentOptions;
	private _store?: Store;
	private _memory?: Memory;
	private _teeClient?: TEEClient;
	/**
	 * Creates an instance of Agent.
	 * @param {AgentOptions} opts - The configuration object for the agent.
	 * @param {string} opts.name - Optional agent name.
	 * @param {string} opts.model - The model used by the agent.
	 * @param {string} opts.preamble - Optional introductory text or context for the agent.
	 * @param {string} [opts.baseUrl] - Optional base URL for API requests.
	 * @param {string} [opts.apiKey] - Optional API key for authentication.
	 * @param {Array<Tool>} [opts.tools] - Optional list of tools available to the agent.
	 * @param {string} [opts.mcpConfigPath] - Optional mcp config path.
	 * @param {Store} [opts.store] - Optional store for the knowledge index.
	 * @param {Memory} [opts.memory] - Optional memory for the agent conversation context.
	 * @param {TEEConfig} [opts.teeConfig] - Optional TEE configuration for secure execution.
	 */
	constructor(opts: AgentOptions) {
		this._opts = opts;
		this._store = opts.store;
		this._memory = opts.memory;

		// Initialize TEE client if configuration is provided
		if (opts.teeConfig) {
			this._teeClient = new TEEClient(opts.teeConfig);
		}

		this._agent = new DelegateAgent(
			opts.name ?? "",
			opts.model,
			opts.apiKey ?? "",
			opts.baseUrl ?? "",
			opts.preamble ?? "",
			opts.mcpConfigPath ?? "",
		);
	}

	/**
	 * Processes a prompt using the agent's tools and model.
	 * @param {string} prompt - The input prompt to process.
	 * @returns {string} - The result of processing the prompt.
	 */
	async prompt(prompt: string): Promise<string> {
		const processPrompt = async (): Promise<string> => {
			// Delegate the prompt processing to the underlying agent and return the result
			const tools = this._opts.tools ?? [];
			const delegateTools: Array<DelegateTool> = [];
			for (const tool of tools) {
				delegateTools.push({
					name: tool.name,
					version: tool.version ?? "",
					description: tool.description,
					parameters: convertParametersToJson(tool.parameters),
					author: tool.author ?? "",
					handler: async (args: string) => {
						const tool_args = JSON.parse(args);
						const args_array = Object.values(tool_args);
						const result = tool.handler(...args_array);
						console.log("asd:", result);
						let result_json;
						if (result instanceof Promise) {
							result_json = JSON.stringify(await result);
							console.log("asd:", result_json);
						} else {
							result_json = JSON.stringify(result);
						}
						return result_json;
					},
				});
			}
			// Sync search documents from the store
			if (this._store) {
				const docs = await this._store.search(prompt);
				prompt = `${prompt}\n\n<attachments>\n${docs.join("")}</attachments>\n`;
			}
			if (this._memory) {
				const result = this._agent.promptWithTools(
					prompt,
					this._memory.messages(),
					delegateTools,
				);
				this._memory.addUserMessage(prompt);
				this._memory.addAIMessage(result);
				return result;
			}
			return this._agent.promptWithTools(prompt, [], delegateTools);
		};

		// Execute with TEE security if configured
		if (this._teeClient) {
			const teeResult = await this._teeClient.executeSecure(processPrompt, {
				attestUserData: `agent-prompt-${this._opts.name || "unknown"}-${Date.now()}`,
				signResult: true,
				includeAttestation: true,
			});

			// Return the result (you can also return the full TEE execution result if needed)
			return teeResult.result;
		}

		// Standard execution without TEE
		return processPrompt();
	}

	/**
	 * Processes a prompt with TEE security guarantees.
	 * @param {string} prompt - The input prompt to process.
	 * @param {object} teeOptions - TEE execution options.
	 * @returns {TEEExecutionResult<string>} - The TEE execution result with verification.
	 */
	async promptSecure(
		prompt: string,
		teeOptions?: {
			attestUserData?: string;
			signResult?: boolean;
			includeAttestation?: boolean;
		},
	): Promise<TEEExecutionResult<string>> {
		if (!this._teeClient) {
			throw new Error(
				"TEE is not configured for this agent. Please provide teeConfig in constructor.",
			);
		}

		const processPrompt = async (): Promise<string> => {
			// Same logic as regular prompt but explicitly within TEE
			const tools = this._opts.tools ?? [];
			const delegateTools: Array<DelegateTool> = [];
			for (const tool of tools) {
				delegateTools.push({
					name: tool.name,
					version: tool.version ?? "",
					description: tool.description,
					parameters: convertParametersToJson(tool.parameters),
					author: tool.author ?? "",
					handler: async (args: string) => {
						const tool_args = JSON.parse(args);
						const args_array = Object.values(tool_args);
						const result = tool.handler(...args_array);
						let result_json;
						if (result instanceof Promise) {
							result_json = JSON.stringify(await result);
						} else {
							result_json = JSON.stringify(result);
						}
						return result_json;
					},
				});
			}

			// Add store search if available
			if (this._store) {
				const docs = await this._store.search(prompt);
				prompt = `${prompt}\n\n<attachments>\n${docs.join("")}</attachments>\n`;
			}

			if (this._memory) {
				const result = this._agent.promptWithTools(
					prompt,
					this._memory.messages(),
					delegateTools,
				);
				this._memory.addUserMessage(prompt);
				this._memory.addAIMessage(result);
				return result;
			}
			return this._agent.promptWithTools(prompt, [], delegateTools);
		};

		return this._teeClient.executeSecure(processPrompt, {
			attestUserData:
				teeOptions?.attestUserData ||
				`agent-secure-prompt-${this._opts.name || "unknown"}-${Date.now()}`,
			signResult: teeOptions?.signResult ?? true,
			includeAttestation: teeOptions?.includeAttestation ?? true,
		});
	}

	/**
	 * Generate TEE attestation for this agent.
	 * @returns {Promise<any>} - TEE attestation proof.
	 */
	async generateAttestation(): Promise<any> {
		if (!this._teeClient) {
			throw new Error(
				"TEE is not configured for this agent. Please provide teeConfig in constructor.",
			);
		}

		return this._teeClient.generateAttestation(
			`agent-${this._opts.name || "unknown"}`,
		);
	}

	/**
	 * Get TEE status for this agent.
	 * @returns {Promise<any>} - TEE status information.
	 */
	async getTEEStatus(): Promise<any> {
		if (!this._teeClient) {
			return { teeEnabled: false, error: "TEE not configured" };
		}

		const status = await this._teeClient.getStatus();
		return { teeEnabled: true, ...status };
	}

	/**
	 * Returns whether TEE is enabled for this agent.
	 * @returns {boolean} - Whether TEE is enabled.
	 */
	isTEEEnabled(): boolean {
		return !!this._teeClient;
	}

	/**
	 * Returns the name of the agent.
	 * @returns {string} - The name of the agent.
	 */
	name(): string {
		return this._opts.name ?? "";
	}

	/**
	 * Returns the model used by the agent.
	 * @returns {string} - The model used by the agent.
	 */
	model(): string {
		return this._opts.model;
	}

	/**
	 * Returns the preamble of the agent.
	 * @returns {string} - The preamble of the agent.
	 */
	preamble(): string | undefined {
		return this._opts.preamble;
	}

	/**
	 * Returns the base url of the agent.
	 * @returns {string} - The base url of the agent.
	 */
	baseUrl(): string | undefined {
		return this._opts.baseUrl;
	}

	/**
	 * Returns the API key of the agent.
	 * @returns {string} - The API key of the agent.
	 */
	apiKey(): string | undefined {
		return this._opts.apiKey;
	}
}

export { Agent, type AgentOptions };
