import { Agent } from '../src/'

const agent = new Agent({
  name: 'A dummy Agent',
  model: 'gpt-4',
  preamble:
    'You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.',
  mcpConfigPath: 'servers_config.json',
})
console.log(agent.prompt('Calculate 10 - 3'))
