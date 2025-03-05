enum MessageType {
  SYSTEM = 'system',
  HUMAN = 'human',
  AI = 'ai',
  TOOL = 'tool',
}

const MessageTypeMap: Record<MessageType, string> = {
  [MessageType.SYSTEM]: 'system',
  [MessageType.HUMAN]: 'user',
  [MessageType.AI]: 'assistant',
  [MessageType.TOOL]: 'tool',
}

class Message {
  content: string
  message_type: MessageType
  id?: string
  tool_calls?: Record<string, any>

  constructor(
    content: string,
    message_type: MessageType = MessageType.SYSTEM,
    id?: string,
    tool_calls?: Record<string, any>,
  ) {
    this.content = content
    this.message_type = message_type
    this.id = id
    this.tool_calls = tool_calls
  }

  static newHumanMessage(content: string): Message {
    return new Message(content, MessageType.HUMAN)
  }

  static newSystemMessage(content: string): Message {
    return new Message(content, MessageType.SYSTEM)
  }

  static newToolMessage(content: string, id: string): Message {
    return new Message(content, MessageType.TOOL, id)
  }

  static newAIMessage(content: string): Message {
    return new Message(content, MessageType.AI)
  }

  withToolCalls(tool_calls: Record<string, any>): this {
    this.tool_calls = tool_calls
    return this
  }

  static messagesFromValue(value: string | object | object[]): Message[] {
    let parsed: object[]
    if (typeof value === 'string') {
      parsed = JSON.parse(value)
    } else if (!Array.isArray(value)) {
      parsed = [value]
    } else {
      parsed = value
    }

    return parsed.map((item: any) => new Message(item.content, item.message_type, item.id, item.tool_calls))
  }

  static messagesToString(messages: Message[]): string {
    return messages.map((msg) => `${MessageTypeMap[msg.message_type]}: ${msg.content}`).join('\n')
  }
}

interface Memory {
  messages(): Message[]
  addUserMessage(message: string): void
  addAIMessage(message: string): void
  addMessage(message: Message): void
  clear(): void
  toString(): string
}

class WindowBufferMemory implements Memory {
  private storage: Message[] = []

  constructor(private windowSize: number = 10) {}

  messages(): Message[] {
    return [...this.storage]
  }

  addUserMessage(message: string): void {
    this.addMessage(Message.newHumanMessage(message))
  }

  addAIMessage(message: string): void {
    this.addMessage(Message.newAIMessage(message))
  }

  addMessage(message: Message): void {
    if (this.storage.length >= this.windowSize) {
      this.storage.shift()
    }
    this.storage.push(message)
  }

  clear(): void {
    this.storage = []
  }

  toString(): string {
    return this.messages()
      .map((msg) => `${MessageTypeMap[msg.message_type]}: ${msg.content}`)
      .join('\n')
  }
}

export { Memory, Message, MessageType, MessageTypeMap, WindowBufferMemory }
