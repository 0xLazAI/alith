import { Agent } from '../agent'

const agent = new Agent({
  name: 'A dummy Agent',
  model: 'deepseek-chat', // or `deepseek-reasoner` for DeepSeek R1
  api_key: '<Your API Key>',
  base_url: 'api.deepseek.com',
  preamble:
    'You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.',
})
console.log(agent.prompt('Calculate 10 - 3'))
