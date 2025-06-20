import type { JSONSchema7 } from "json-schema";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

/**
 * The Parameters type, which can be one of the following three types:
 * 1. string: Directly represents a JSONSchema string
 * 2. z.ZodTypeAny: A type-safe schema defined using Zod
 * 3. JSONSchema7: A standard JSON Schema object
 */
type Parameters = string | z.ZodTypeAny | JSONSchema7;

/**
 * Represents the structure of a tool or function that can be used in various contexts.
 * This structure is designed to be flexible and type-safe, allowing for different types of parameter definitions.
 * It can be used to define tools in a modular and reusable way.
 */
type Tool = {
  /**
   * The name of the tool. This should be a unique identifier for the tool.
   */
  name: string;

  /**
   * A human-readable description of the tool. This is useful for documentation and user interfaces.
   */
  description: string;

  /**
   * The parameter definition of the tool. This can be one of the following:
   * - A JSON string representing the parameters.
   * - A Zod schema defining the parameters in a type-safe manner.
   * - A JSON Schema object defining the parameters.
   * This flexibility allows the tool to be defined in different ways, depending on the use case.
   */
  parameters: Parameters;

  /**
   * The version number of the tool (optional). This can be used to track different versions of the tool.
   */
  version?: string;

  /**
   * The author of the tool (optional). This can be used to credit the creator of the tool.
   */
  author?: string;

  /**
   * The handler function of the tool. This function is called when the tool is executed.
   * It accepts any number of arguments and returns any type of result.
   * The arguments passed to this function should match the parameters defined in the `parameters` field.
   */
  handler: (...args: unknown[]) => unknown;
};

/**
 * Converts the Parameters type value to a JSON string.
 * @param parameters - The parameter definition, which can be a string, Zod schema, or JSON Schema object.
 * @returns The converted JSON string.
 * @throws If the parameter type is invalid, an error is thrown.
 */
function convertParametersToJson(parameters: Parameters): string {
  // If parameters is a string, return it directly
  if (typeof parameters === "string") {
    return parameters;
  }
  // If parameters is a Zod schema
  if (parameters instanceof z.ZodSchema) {
    // Use zodToJsonSchema to convert the Zod schema to JSON Schema
    // target: 'jsonSchema7' specifies the target format as JSON Schema 7
    const jsonSchema = zodToJsonSchema(parameters, {
      target: "jsonSchema7",
    });
    // Convert the JSON Schema object to a JSON string
    return JSON.stringify(jsonSchema);
  }
  // If parameters is an object (assumed to be a JSON Schema object)
  if (typeof parameters === "object") {
    // Directly convert the object to a JSON string
    return JSON.stringify(parameters);
  }
  // If the type of parameters does not match any of the above, throw an error
  throw new Error("Invalid parameters type");
}

// Export the function and types for use in other modules
export { convertParametersToJson, type Parameters, type Tool };
