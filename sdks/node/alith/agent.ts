import { DelegateAgent, DelegateTool } from '../index.js'

// Represents an agent that can process prompts using tools
class Agent {
  /**
   * Creates an instance of Agent.
   * @param {string} name - The name of the agent.
   * @param {string} model - The model used by the agent.
   * @param {Array<DelegateTool>} tools - A list of tools available to the agent.
   */
  private _agent: DelegateAgent
  constructor(public name: string, public model: string, public tools: Array<DelegateTool> = []) {
    this._agent = new DelegateAgent(this.name, this.model, this.tools)
  }

  /**
   * Processes a prompt using the agent's tools and model.
   * @param {string} prompt - The input prompt to process.
   * @returns {string} - The result of processing the prompt.
   */
  prompt(prompt: string): string {
    // Process the prompt and return the result
    return this._agent.prompt(prompt)
  }
}

export { Agent }
