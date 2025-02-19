import { DelegateAgent, type DelegateTool } from './internal'
import { type Tool, convertParametersToJson } from './tool'

// Define the configuration structure for an Agent
type AgentOptions = {
  name: string // The name of the agent
  model: string // The model used by the agent
  preamble: string // Introductory text or context for the agent
  baseUrl?: string // Optional base URL for API requests
  apiKey?: string // Optional API key for authentication
  tools?: Array<Tool> // Optional list of tools available to the agent
}

// Represents an agent that can process prompts using tools
class Agent {
  private _agent: DelegateAgent
  private _opts: AgentOptions
  /**
   * Creates an instance of Agent.
   * @param {AgentOptions} opts - The configuration object for the agent.
   * @param {string} opts.name - The name of the agent.
   * @param {string} opts.model - The model used by the agent.
   * @param {string} opts.preamble - Introductory text or context for the agent.
   * @param {string} [opts.baseUrl] - Optional base URL for API requests.
   * @param {string} [opts.apiKey] - Optional API key for authentication.
   * @param {Array<Tool>} [opts.tools] - Optional list of tools available to the agent.
   */
  constructor(opts: AgentOptions) {
    this._opts = opts
    this._agent = new DelegateAgent(opts.name, opts.model, opts.apiKey ?? '', opts.baseUrl ?? '', opts.preamble ?? '')
  }

  /**
   * Processes a prompt using the agent's tools and model.
   * @param {string} prompt - The input prompt to process.
   * @returns {string} - The result of processing the prompt.
   */
  prompt(prompt: string): string {
    // Delegate the prompt processing to the underlying agent and return the result
    const tools = this._opts.tools ?? []
    const delegateTools: Array<DelegateTool> = []
    for (const tool of tools) {
      delegateTools.push({
        name: tool.name,
        version: tool.version ?? '',
        description: tool.description,
        parameters: convertParametersToJson(tool.parameters),
        author: tool.author ?? '',
        handler: (args: string) => {
          const tool_args = JSON.parse(args)
          const args_array = Object.values(tool_args)
          return tool.handler(...args_array)
        },
      })
    }
    return this._agent.promptWithTools(prompt, delegateTools)
  }

  /**
   * Returns the name of the agent.
   * @returns {string} - The name of the agent.
   */
  name(): string {
    return this._opts.name
  }

  /**
   * Returns the model used by the agent.
   * @returns {string} - The model used by the agent.
   */
  model(): string {
    return this._opts.model
  }
}

export { Agent, type AgentOptions }
