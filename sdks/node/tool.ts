type Tool = {
  name: string
  description: string
  parameters: string
  version?: string
  author?: string
  handler: (...args: any[]) => any
}

export { Tool }
