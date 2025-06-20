import { z } from "zod";
import { Agent } from "./agent";

export class Extractor {
  constructor(public agent: Agent, public model: z.ZodSchema) {}

  async extract(input: string): Promise<z.infer<typeof this.model>> {
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
          name: "extractor",
          description: "Extract the data structure from the input string.",
          parameters: this.model,
          handler: (...args: any[]) => parseArgs(this.model, ...args),
        },
      ],
    });
    const result = await agent.prompt(input);
    return this.model.parse(JSON.parse(result));
  }
}

export function parseArgs<TActionSchema extends z.ZodTypeAny = z.ZodTypeAny>(
  argsSchema: TActionSchema,
  ...args: any[]
): z.infer<TActionSchema> {
  // If the schema is an object, parse the arguments into an object
  if (isZodObject(argsSchema)) {
    const properties = argsSchema.shape;
    const argsObject: Record<string, any> = {};
    let index = 0;

    for (const key in properties) {
      if (properties.hasOwnProperty(key)) {
        if (index >= args.length) {
          const defaultValue = getDefaultValue(properties[key], args[index]);
          argsObject[key] = defaultValue;
        } else {
          argsObject[key] = args[index];
        }
        index++;
      }
    }
    return argsSchema.parse(argsObject);
  }
  // If the schema is not an object, parse the arguments directly
  return argsSchema.parse(args[0]);
}

function isZodObject(schema: any): schema is z.ZodObject<Record<string, any>> {
  return (
    schema instanceof z.ZodObject ||
    (schema?._def && schema._def.typeName === "ZodObject") ||
    (typeof schema.shape === "function" &&
      schema.shape().constructor.name === "Object")
  );
}

function getDefaultValue(zodType: any, value: any): any {
  if (zodType instanceof z.ZodString) {
    return "";
  }
  if (zodType instanceof z.ZodNumber) {
    return 0;
  }
  if (zodType instanceof z.ZodBoolean) {
    return false;
  }
  return parseArgs(zodType, value);
}
