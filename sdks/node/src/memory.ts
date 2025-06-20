import { Message } from "./internal";

class MessageBuilder {
  static newHumanMessage(content: string): Message {
    return { role: "user", content };
  }

  static newSystemMessage(content: string): Message {
    return { role: "system", content };
  }

  static newToolMessage(content: string): Message {
    return { role: "tool", content };
  }

  static newAIMessage(content: string): Message {
    return { role: "assistant", content };
  }

  static messagesFromValue(value: string | object | object[]): Message[] {
    let parsed: object[];
    if (typeof value === "string") {
      parsed = JSON.parse(value);
    } else if (!Array.isArray(value)) {
      parsed = [value];
    } else {
      parsed = value;
    }

    return parsed.map((item: any) => {
      return { role: item.role, content: item.content };
    });
  }

  static messagesToString(messages: Message[]): string {
    return messages.map((msg) => `${msg.role}: ${msg.content}`).join("\n");
  }
}

interface Memory {
  messages(): Message[];
  addUserMessage(message: string): void;
  addAIMessage(message: string): void;
  addMessage(message: Message): void;
  clear(): void;
  toString(): string;
}

class WindowBufferMemory implements Memory {
  private storage: Message[] = [];

  constructor(private windowSize = 10) {}

  messages(): Message[] {
    return [...this.storage];
  }

  addUserMessage(message: string): void {
    this.addMessage(MessageBuilder.newHumanMessage(message));
  }

  addAIMessage(message: string): void {
    this.addMessage(MessageBuilder.newAIMessage(message));
  }

  addMessage(message: Message): void {
    if (this.storage.length >= this.windowSize) {
      this.storage.shift();
    }
    this.storage.push(message);
  }

  clear(): void {
    this.storage = [];
  }

  toString(): string {
    return this.messages()
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n");
  }
}

export { type Memory, Message, MessageBuilder, WindowBufferMemory };
