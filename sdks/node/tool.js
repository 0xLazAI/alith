'use strict'
Object.defineProperty(exports, '__esModule', { value: true })
exports.convertParametersToJson = convertParametersToJson
const zod_to_json_schema_1 = require('zod-to-json-schema')
const zod_1 = require('zod')
/**
 * Converts the Parameters type value to a JSON string.
 * @param parameters - The parameter definition, which can be a string, Zod schema, or JSON Schema object.
 * @returns The converted JSON string.
 * @throws If the parameter type is invalid, an error is thrown.
 */
function convertParametersToJson(parameters) {
  // If parameters is a string, return it directly
  if (typeof parameters === 'string') {
    return parameters
  }
  // If parameters is a Zod schema
  else if (parameters instanceof zod_1.z.ZodSchema) {
    // Use zodToJsonSchema to convert the Zod schema to JSON Schema
    // target: 'jsonSchema7' specifies the target format as JSON Schema 7
    const jsonSchema = (0, zod_to_json_schema_1.zodToJsonSchema)(parameters, {
      target: 'jsonSchema7',
    })
    // Convert the JSON Schema object to a JSON string
    return JSON.stringify(jsonSchema)
  }
  // If parameters is an object (assumed to be a JSON Schema object)
  else if (typeof parameters === 'object') {
    // Directly convert the object to a JSON string
    return JSON.stringify(parameters)
  } else {
    // If the type of parameters does not match any of the above, throw an error
    throw new Error('Invalid parameters type')
  }
}
