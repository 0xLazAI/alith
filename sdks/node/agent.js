'use strict'
Object.defineProperty(exports, '__esModule', { value: true })
exports.Agent = void 0
const internal_1 = require('./internal')
const tool_1 = require('./tool')
// Represents an agent that can process prompts using tools
class Agent {
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
  constructor(opts) {
    this._opts = opts
    this._agent = new internal_1.DelegateAgent(
      opts.name,
      opts.model,
      opts.api_key ?? '',
      opts.base_url ?? '',
      opts.preamble ?? '',
    )
  }
  /**
   * Processes a prompt using the agent's tools and model.
   * @param {string} prompt - The input prompt to process.
   * @returns {string} - The result of processing the prompt.
   */
  prompt(prompt) {
    // Delegate the prompt processing to the underlying agent and return the result
    let tools = this._opts.tools ?? []
    let delegate_tools = []
    for (let tool of tools) {
      delegate_tools.push({
        name: tool.name,
        version: tool.version ?? '',
        description: tool.description,
        parameters: (0, tool_1.convertParametersToJson)(tool.parameters),
        author: tool.author ?? '',
        handler: (args) => {
          let tool_args = JSON.parse(args)
          const args_array = Object.values(tool_args)
          return tool.handler(...args_array)
        },
      })
    }
    return this._agent.promptWithTools(prompt, delegate_tools)
  }
  /**
   * Returns the name of the agent.
   * @returns {string} - The name of the agent.
   */
  name() {
    return this._opts.name
  }
  /**
   * Returns the model used by the agent.
   * @returns {string} - The model used by the agent.
   */
  model() {
    return this._opts.model
  }
}
exports.Agent = Agent
