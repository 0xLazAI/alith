// sdks/go/examples/simple_test.go
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

	// Create agent without tools first
	agent := alith.NewAgent("Simple Assistant", "gpt-4")
	agent.WithCredentials(apiKey, "https://api.openai.com/v1")
	agent.WithPreamble("You are a helpful assistant.")

	// Initialize
	if err := agent.Initialize(); err != nil {
		log.Fatal("Failed to initialize agent:", err)
	}

	// Use the agent
	response, err := agent.Prompt("Say hello!")
	if err != nil {
		log.Fatal("Error:", err)
	}

	fmt.Println("Agent response:", response)
}
