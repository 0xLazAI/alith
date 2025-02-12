import { JSONSchema7 } from 'json-schema'
import { z } from 'zod'
/**
 * The Parameters type, which can be one of the following three types:
 * 1. string: Directly represents a JSONSchema string
 * 2. z.ZodTypeAny: A type-safe schema defined using Zod
 * 3. JSONSchema7: A standard JSON Schema object
 */
type Parameters = string | z.ZodTypeAny | JSONSchema7
/**
 * Represents the structure of a tool or function that can be used in various contexts.
 * This structure is designed to be flexible and type-safe, allowing for different types of parameter definitions.
 * It can be used to define tools in a modular and reusable way.
 */
type Tool = {
  /**
   * The name of the tool. This should be a unique identifier for the tool.
   */
  name: string
  /**
   * A human-readable description of the tool. This is useful for documentation and user interfaces.
   */
  description: string
  /**
   * The parameter definition of the tool. This can be one of the following:
   * - A JSON string representing the parameters.
   * - A Zod schema defining the parameters in a type-safe manner.
   * - A JSON Schema object defining the parameters.
   * This flexibility allows the tool to be defined in different ways, depending on the use case.
   */
  parameters: Parameters
  /**
   * The version number of the tool (optional). This can be used to track different versions of the tool.
   */
  version?: string
  /**
   * The author of the tool (optional). This can be used to credit the creator of the tool.
   */
  author?: string
  /**
   * The handler function of the tool. This function is called when the tool is executed.
   * It accepts any number of arguments and returns any type of result.
   * The arguments passed to this function should match the parameters defined in the `parameters` field.
   */
  handler: (...args: any[]) => any
}
/**
 * Converts the Parameters type value to a JSON string.
 * @param parameters - The parameter definition, which can be a string, Zod schema, or JSON Schema object.
 * @returns The converted JSON string.
 * @throws If the parameter type is invalid, an error is thrown.
 */
declare function convertParametersToJson(parameters: Parameters): string
export { convertParametersToJson, Parameters, Tool }
