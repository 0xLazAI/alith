# Quick Start Guide - Alith Go SDK

## Summary

The Alith Go SDK is now set up and ready to use! The SDK allows you to create AI agents with custom tools using Go, powered by the Rust-based Alith framework.

## What's Included

✅ **Go SDK Package** (`sdks/go/alith/`)
- Agent creation and management
- Tool registration with safe callbacks
- Support for multiple LLM providers

✅ **Example Code** (`sdks/go/examples/`)
- `simple_agent.go` - Weather assistant with mock tool
- `simple_example.go` - Basic agent without tools

✅ **Pre-built Library**
- Rust library compiled for GNU compatibility
- Located at `sdks/go/alith/libalith_go_sdk.a`

## How to Run the Examples

### Step 1: Set Your OpenAI API Key

```powershell
# In PowerShell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

### Step 2: Run the Example

**Option A: Using the provided script**
```powershell
cd sdks\go
.\run_example.ps1
```

**Option B: Manually**
```powershell
cd sdks\go\examples
go run simple_agent.go
```

## What the Example Does

The `simple_agent.go` example:

1. Creates a weather assistant agent
2. Registers a mock weather tool that returns:
   - Location
   - Temperature: 22°C
   - Condition: Sunny

3. Asks the agent: "What's the weather like in Tokyo?"
4. The agent will use the tool and respond with the weather information

## Expected Output

The agent will:
- Detect that it needs to use the `get_weather` tool
- Call the tool with `{"location": "Tokyo"}`
- Receive the mock weather data
- Respond with something like: "The weather in Tokyo is currently sunny with a temperature of 22°C."

## Troubleshooting

### Error: "OPENAI_API_KEY is not set"
- Make sure to set the environment variable before running
- Or pass it to the script: `.\run_example.ps1 -ApiKey "your-key"`

### Linking Errors
- The library must be compiled with the GNU target: `x86_64-pc-windows-gnu`
- Make sure `libalith_go_sdk.a` exists in `sdks/go/alith/`

### API Errors
- Check that your API key is valid
- Ensure you have credits in your OpenAI account
- Model "gpt-4" requires appropriate access level

## Customizing the Example

### Using a Different Model

Change the model name:
```go
agent := alith.NewAgent("My Assistant", "gpt-3.5-turbo")  // or any other model
```

### Adding Your Own Tools

Define a handler function:
```go
func myTool(params map[string]interface{}) (interface{}, error) {
    // Your tool logic here
    return result, nil
}
```

Register it:
```go
tool := alith.Tool{
    Name:        "my_tool",
    Description: "What the tool does",
    Parameters:  /* JSON schema */,
    Handler:     myTool,
}
agent.WithTool(tool)
```

## Next Steps

- Implement real weather API integration
- Add database tools
- Create web search tools
- Build multi-agent systems

## Support

For issues or questions:
- Check the main README: `sdks/go/README.md`
- Review examples: `sdks/go/examples/`
- See the main Alith documentation

Happy building! 🚀

