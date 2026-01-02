// sdks/go/examples/weather_demo.go
package main

import (
	"fmt"
	"log"

	"github.com/0xLazAI/alith/sdks/go/alith"
)

// Mock weather tool - returns fake weather data
func weatherTool(params map[string]interface{}) (interface{}, error) {
	location, ok := params["location"].(string)
	if !ok {
		return nil, fmt.Errorf("location parameter required")
	}

	// Different mock data for different cities
	weatherData := map[string]map[string]string{
		"Tokyo": {
			"location":    "Tokyo",
			"temperature": "22°C",
			"condition":   "Sunny",
			"humidity":    "65%",
		},
		"London": {
			"location":    "London",
			"temperature": "12°C",
			"condition":   "Rainy",
			"humidity":    "85%",
		},
		"New York": {
			"location":    "New York",
			"temperature": "18°C",
			"condition":   "Cloudy",
			"humidity":    "70%",
		},
	}

	// Return data for the requested city, or default for unknown cities
	if data, exists := weatherData[location]; exists {
		return data, nil
	}

	return map[string]string{
		"location":    location,
		"temperature": "20°C",
		"condition":   "Clear",
		"humidity":    "60%",
	}, nil
}

// Mock stock price tool
func stockPriceTool(params map[string]interface{}) (interface{}, error) {
	symbol, ok := params["symbol"].(string)
	if !ok {
		return nil, fmt.Errorf("symbol parameter required")
	}

	// Mock stock prices
	prices := map[string]float64{
		"AAPL":  175.43,
		"GOOGL": 142.56,
		"TSLA":  248.92,
		"MSFT":  378.91,
	}

	if price, exists := prices[symbol]; exists {
		return map[string]interface{}{
			"symbol": symbol,
			"price":  price,
			"change": "+2.5%",
		}, nil
	}

	return map[string]interface{}{
		"symbol": symbol,
		"price":  100.00,
		"change": "0%",
	}, nil
}

func main() {
	// Using Groq API
	apiKey := "<YOUR_GROQ_API_KEY>"

	// Create agent with multiple tools
	agent := alith.NewAgent("Multi-Tool Assistant", "llama-3.3-70b-versatile")
	agent.WithCredentials(apiKey, "https://api.groq.com/openai/v1")
	agent.WithPreamble("You are a helpful assistant with access to weather and stock price information. Provide friendly, informative responses.")

	// Add weather tool
	weatherToolDef := alith.Tool{
		Name:        "get_weather",
		Description: "Get current weather information for a specific city",
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

	// Add stock price tool
	stockToolDef := alith.Tool{
		Name:        "get_stock_price",
		Description: "Get current stock price for a given ticker symbol",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"symbol": map[string]interface{}{
					"type":        "string",
					"description": "Stock ticker symbol (e.g., AAPL, GOOGL)",
				},
			},
			"required": []string{"symbol"},
		},
		Handler: stockPriceTool,
	}

	agent.WithTool(weatherToolDef)
	agent.WithTool(stockToolDef)

	// Initialize
	if err := agent.Initialize(); err != nil {
		log.Fatal("Failed to initialize agent:", err)
	}

	// Test various prompts
	prompts := []string{
		"What's the weather like in Tokyo?",
		"How's the weather in London?",
		"What's the stock price of Apple (AAPL)?",
		"Tell me about the weather in Paris",
	}

	for i, prompt := range prompts {
		fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("Query %d: %s\n", i+1, prompt)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		response, err := agent.Prompt(prompt)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("Response: %s\n", response)
	}

	fmt.Printf("\n✅ All demos completed successfully!\n")
}
