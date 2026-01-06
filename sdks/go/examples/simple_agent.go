// sdks/go/examples/simple_agent.go
package main

import (
	"fmt"
	"log"

	"github.com/0xLazAI/alith/sdks/go/alith"
)

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
	// Using Groq API
	apiKey := "<YOUR_GROQ_API_KEY>"

	// Create agent with Groq's Llama model
	agent := alith.NewAgent("Weather Assistant", "llama-3.3-70b-versatile")
	agent.WithCredentials(apiKey, "https://api.groq.com/openai/v1")
	agent.WithPreamble("You are a helpful weather assistant. When you use the get_weather tool, provide a friendly response based on the weather data.")

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
