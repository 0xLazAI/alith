import { Tool } from './tool'
type AgentOptions = {
  name: string
  model: string
  preamble: string
  base_url?: string
  api_key?: string
  tools?: Array<Tool>
}
declare class Agent {
  private _agent
  private _opts
  /**
   * Creates an instance of Agent.
   * @param {AgentOptions} opts - The configuration object for the agent.
   * @param {string} opts.name - The name of the agent.
   * @param {string} opts.model - The model used by the agent.
   * @param {string} opts.preamble - Introductory text or context for the agent.
   * @param {string} [opts.base_url] - Optional base URL for API requests.
   * @param {string} [opts.api_key] - Optional API key for authentication.
   * @param {Array<Tool>} [opts.tools] - Optional list of tools available to the agent.
   */
  constructor(opts: AgentOptions)
  /**
   * Processes a prompt using the agent's tools and model.
   * @param {string} prompt - The input prompt to process.
   * @returns {string} - The result of processing the prompt.
   */
  prompt(prompt: string): string
  /**
   * Returns the name of the agent.
   * @returns {string} - The name of the agent.
   */
  name(): string
  /**
   * Returns the model used by the agent.
   * @returns {string} - The model used by the agent.
   */
  model(): string
}
export { Agent, AgentOptions }
