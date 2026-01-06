# Alith Go SDK

Go bindings for the Alith AI agent framework.

## Prerequisites

- Go 1.21 or later
- Rust toolchain with `x86_64-pc-windows-gnu` target (Windows) or appropriate GNU target for your platform
- GCC (MinGW on Windows)

## Building

### 1. Install Rust GNU target (Windows)

```bash
rustup target add x86_64-pc-windows-gnu
```

### 2. Build the Rust library

```bash
cargo build --release --target x86_64-pc-windows-gnu -p alith-go-sdk
```

### 3. Copy the library to the alith package

```bash
# Windows
copy target\x86_64-pc-windows-gnu\release\libalith_go_sdk.a sdks\go\alith\libalith_go_sdk.a

# Linux/macOS (adjust target as needed)
cp target/x86_64-unknown-linux-gnu/release/libalith_go_sdk.a sdks/go/alith/libalith_go_sdk.a
```

## Usage

### Basic Example

```go
package main

import (
    "fmt"
    "log"
    "os"
    
    "github.com/0xLazAI/alith/sdks/go/alith"
)

func main() {
    // Get API key from environment
    apiKey := os.Getenv("OPENAI_API_KEY")
    if apiKey == "" {
        log.Fatal("Please set OPENAI_API_KEY environment variable")
    }

    // Create agent
    agent := alith.NewAgent("My Assistant", "gpt-4")
    agent.WithCredentials(apiKey, "https://api.openai.com/v1")
    agent.WithPreamble("You are a helpful assistant.")

    // Initialize
    if err := agent.Initialize(); err != nil {
        log.Fatal("Failed to initialize agent:", err)
    }

    // Use the agent
    response, err := agent.Prompt("Hello!")
    if err != nil {
        log.Fatal("Error:", err)
    }

    fmt.Println("Response:", response)
}
```

### Example with Tools

```go
package main

import (
    "fmt"
    "log"
    "os"
    
    "github.com/0xLazAI/alith/sdks/go/alith"
)

// Define a tool handler
func weatherTool(params map[string]interface{}) (interface{}, error) {
    location, ok := params["location"].(string)
    if !ok {
        return nil, fmt.Errorf("location parameter required")
    }

    // Mock weather data
    return map[string]interface{}{
        "location":    location,
        "temperature": "22°C",
        "condition":   "Sunny",
    }, nil
}

func main() {
    apiKey := os.Getenv("OPENAI_API_KEY")
    if apiKey == "" {
        log.Fatal("Please set OPENAI_API_KEY environment variable")
    }

    // Create agent
    agent := alith.NewAgent("Weather Assistant", "gpt-4")
    agent.WithCredentials(apiKey, "https://api.openai.com/v1")
    agent.WithPreamble("You are a helpful weather assistant.")

    // Add weather tool
    weatherToolDef := alith.Tool{
        Name:        "get_weather",
        Description: "Get current weather for a location",
        Parameters: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "location": map[string]interface{}{
                    "type":        "string",
                    "description": "The city name",
                },
            },
            "required": []string{"location"},
        },
        Handler: weatherTool,
    }

    agent.WithTool(weatherToolDef)

    // Initialize
    if err := agent.Initialize(); err != nil {
        log.Fatal("Failed to initialize agent:", err)
    }

    // Use the agent
    response, err := agent.Prompt("What's the weather like in Tokyo?")
    if err != nil {
        log.Fatal("Error:", err)
    }

    fmt.Println("Agent response:", response)
}
```

## Running the Examples

### Set your API key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key-here"

# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"
```

### Run the example

```bash
cd examples
go run simple_agent.go
```

## Architecture

The Go SDK uses CGo to interact with the Rust library. Tool handlers are registered in Go and called back from Rust using a callback ID system to ensure memory safety and proper garbage collection.

## Supported Models

The SDK supports various LLM providers:

- OpenAI (gpt-4, gpt-3.5-turbo, etc.)
- Anthropic (claude-3-opus, claude-3-sonnet, etc.)
- Perplexity (llama models)
- Any OpenAI-compatible API

## License

Apache-2.0

