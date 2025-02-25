import { Agent } from '../agent'

const agent = new Agent({
  name: 'A dummy Agent',
  model: 'grok-3',
  api_key: '<Your API Key>',
  base_url: 'api.grok.ai/v1',
  preamble:
    'You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.',
})
console.log(agent.prompt('Calculate 10 - 3'))
