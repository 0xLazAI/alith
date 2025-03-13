import { z } from 'zod'
import { Agent } from './agent'

export class Extractor {
  constructor(public agent: Agent, public model: z.ZodSchema) {}

  extract(input: string): z.infer<typeof this.model> {
    const agent = new Agent({
      name: this.agent.name(),
      model: this.agent.model(),
      preamble: `Extract the data structure from the input string.
Note you MUST use the tool named 'extractor' to extract the input string to the
data structure.
        `,
      baseUrl: this.agent.baseUrl(),
      apiKey: this.agent.apiKey(),
      tools: [
        {
          name: 'extractor',
          description: 'Extract the data structure from the input string.',
          parameters: this.model,
          handler: (...args: any[]) => parseArgs(this.model, ...args),
        },
      ],
    })
    const result = agent.prompt(input)
    return this.model.parse(JSON.parse(result))
  }
}

export function parseArgs<TActionSchema extends z.ZodTypeAny = z.ZodTypeAny>(
  argsSchema: TActionSchema,
  ...args: any[]
): z.infer<TActionSchema> {
  // If the schema is not an object, parse the arguments directly
  if (!(argsSchema instanceof z.ZodObject)) {
    return argsSchema.parse(args[0]) // Assume the first argument is the value
  }

  // If the schema is an object, parse the arguments into an object
  const properties = argsSchema.shape
  const argsObject: Record<string, any> = {}
  let index = 0
  for (const key in properties) {
    if (properties.hasOwnProperty(key)) {
      const value =
        args[index] !== undefined
          ? args[index]
          : properties[key] instanceof z.ZodString
          ? ''
          : properties[key] instanceof z.ZodNumber
          ? 0
          : properties[key] instanceof z.ZodBoolean
          ? false
          : null
      argsObject[key] = value
      index++
    }
  }
  return argsSchema.parse(argsObject)
}
