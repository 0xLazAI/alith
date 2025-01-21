import { Agent } from '../agent'

const agent = new Agent({
  name: 'A dummy Agent',
  model: 'gpt-4',
  preamble: 'You are a comedian here to entertain the user using humour and jokes.',
})
console.log(agent.prompt('Entertain me!'))
