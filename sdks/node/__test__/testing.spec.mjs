import test from 'ava'

import { DelegateAgent } from '../index.js'

test('test', (t) => {
  const result = new DelegateAgent('dummy', 'gpt4o-mini', [])
  t.is(result.model, 'gpt4o-mini')
})
