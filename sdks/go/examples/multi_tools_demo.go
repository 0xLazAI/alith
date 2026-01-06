// sdks/go/examples/multi_tools_demo.go
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/0xLazAI/alith/sdks/go/alith"
)

// Calculator tool - performs basic math operations
func calculatorTool(params map[string]interface{}) (interface{}, error) {
	operation, ok := params["operation"].(string)
	if !ok {
		return nil, fmt.Errorf("operation parameter required")
	}

	a, ok := params["a"].(float64)
	if !ok {
		return nil, fmt.Errorf("parameter 'a' must be a number")
	}

	b, ok := params["b"].(float64)
	if !ok {
		return nil, fmt.Errorf("parameter 'b' must be a number")
	}

	var result float64
	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		result = a / b
	default:
		return nil, fmt.Errorf("unsupported operation: %s", operation)
	}

	return map[string]interface{}{
		"operation": operation,
		"a":         a,
		"b":         b,
		"result":    result,
	}, nil
}

// Random number generator tool
func randomTool(params map[string]interface{}) (interface{}, error) {
	min, ok := params["min"].(float64)
	if !ok {
		min = 1
	}

	max, ok := params["max"].(float64)
	if !ok {
		max = 100
	}

	if min > max {
		return nil, fmt.Errorf("min cannot be greater than max")
	}

	rand.Seed(time.Now().UnixNano())
	result := min + rand.Float64()*(max-min)

	return map[string]interface{}{
		"min":    min,
		"max":    max,
		"result": result,
	}, nil
}

// Text analyzer tool - analyzes text properties
func textAnalyzerTool(params map[string]interface{}) (interface{}, error) {
	text, ok := params["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text parameter required")
	}

	words := len([]rune(text))
	chars := len(text)
	lines := 1
	for _, char := range text {
		if char == '\n' {
			lines++
		}
	}

	// Count vowels
	vowels := 0
	for _, char := range text {
		switch char {
		case 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U':
			vowels++
		}
	}

	return map[string]interface{}{
		"text":        text,
		"word_count":  words,
		"char_count":  chars,
		"line_count":  lines,
		"vowel_count": vowels,
	}, nil
}

// Database simulator tool - simulates database operations
func databaseTool(params map[string]interface{}) (interface{}, error) {
	action, ok := params["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action parameter required")
	}

	// Mock database
	users := map[string]map[string]interface{}{
		"1": {"id": "1", "name": "Alice", "email": "alice@example.com", "age": 25},
		"2": {"id": "2", "name": "Bob", "email": "bob@example.com", "age": 30},
		"3": {"id": "3", "name": "Charlie", "email": "charlie@example.com", "age": 35},
	}

	switch action {
	case "list_users":
		var userList []map[string]interface{}
		for _, user := range users {
			userList = append(userList, user)
		}
		return map[string]interface{}{
			"action": "list_users",
			"users":  userList,
			"count":  len(users),
		}, nil

	case "get_user":
		id, ok := params["id"].(string)
		if !ok {
			return nil, fmt.Errorf("id parameter required for get_user")
		}

		if user, exists := users[id]; exists {
			return map[string]interface{}{
				"action": "get_user",
				"user":   user,
			}, nil
		}
		return nil, fmt.Errorf("user not found")

	case "search_users":
		query, ok := params["query"].(string)
		if !ok {
			return nil, fmt.Errorf("query parameter required for search_users")
		}

		var results []map[string]interface{}
		for _, user := range users {
			if name, ok := user["name"].(string); ok {
				if len(query) > 0 && len(name) >= len(query) {
					// Simple substring search
					found := false
					for i := 0; i <= len(name)-len(query); i++ {
						if name[i:i+len(query)] == query {
							found = true
							break
						}
					}
					if found {
						results = append(results, user)
					}
				}
			}
		}

		return map[string]interface{}{
			"action":  "search_users",
			"query":   query,
			"results": results,
			"count":   len(results),
		}, nil

	default:
		return nil, fmt.Errorf("unsupported action: %s", action)
	}
}

// File system simulator tool
func fileSystemTool(params map[string]interface{}) (interface{}, error) {
	action, ok := params["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action parameter required")
	}

	// Mock file system
	files := map[string]map[string]interface{}{
		"/home/user/documents/report.txt": {
			"path":    "/home/user/documents/report.txt",
			"size":    1024,
			"type":    "file",
			"created": "2024-01-15",
		},
		"/home/user/images/photo.jpg": {
			"path":    "/home/user/images/photo.jpg",
			"size":    2048000,
			"type":    "file",
			"created": "2024-01-20",
		},
		"/home/user/documents/": {
			"path":    "/home/user/documents/",
			"size":    0,
			"type":    "directory",
			"created": "2024-01-10",
		},
	}

	switch action {
	case "list_files":
		path, ok := params["path"].(string)
		if !ok {
			path = "/home/user/"
		}

		var fileList []map[string]interface{}
		for filePath, info := range files {
			if len(filePath) >= len(path) && filePath[:len(path)] == path {
				fileList = append(fileList, info)
			}
		}

		return map[string]interface{}{
			"action": "list_files",
			"path":   path,
			"files":  fileList,
			"count":  len(fileList),
		}, nil

	case "file_info":
		path, ok := params["path"].(string)
		if !ok {
			return nil, fmt.Errorf("path parameter required for file_info")
		}

		if info, exists := files[path]; exists {
			return map[string]interface{}{
				"action": "file_info",
				"info":   info,
			}, nil
		}
		return nil, fmt.Errorf("file not found")

	default:
		return nil, fmt.Errorf("unsupported action: %s", action)
	}
}

func main() {
	// Using Groq API
	apiKey := "<YOUR_GROQ_API_KEY>"

	// Create agent with multiple tools
	agent := alith.NewAgent("Multi-Tool Assistant", "llama-3.3-70b-versatile")
	agent.WithCredentials(apiKey, "https://api.groq.com/openai/v1")
	agent.WithPreamble("You are a helpful assistant with access to various tools: calculator, random number generator, text analyzer, database operations, and file system operations. Use the appropriate tool for each request.")

	// Calculator tool
	calcTool := alith.Tool{
		Name:        "calculator",
		Description: "Perform basic mathematical operations (add, subtract, multiply, divide)",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"operation": map[string]interface{}{
					"type":        "string",
					"description": "The operation to perform (add, subtract, multiply, divide)",
				},
				"a": map[string]interface{}{
					"type":        "number",
					"description": "First number",
				},
				"b": map[string]interface{}{
					"type":        "number",
					"description": "Second number",
				},
			},
			"required": []string{"operation", "a", "b"},
		},
		Handler: calculatorTool,
	}

	// Random number tool
	randomTool := alith.Tool{
		Name:        "random_number",
		Description: "Generate a random number between min and max values",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"min": map[string]interface{}{
					"type":        "number",
					"description": "Minimum value (default: 1)",
				},
				"max": map[string]interface{}{
					"type":        "number",
					"description": "Maximum value (default: 100)",
				},
			},
		},
		Handler: randomTool,
	}

	// Text analyzer tool
	textTool := alith.Tool{
		Name:        "text_analyzer",
		Description: "Analyze text properties like word count, character count, etc.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"text": map[string]interface{}{
					"type":        "string",
					"description": "The text to analyze",
				},
			},
			"required": []string{"text"},
		},
		Handler: textAnalyzerTool,
	}

	// Database tool
	dbTool := alith.Tool{
		Name:        "database",
		Description: "Perform database operations (list_users, get_user, search_users)",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action": map[string]interface{}{
					"type":        "string",
					"description": "The action to perform (list_users, get_user, search_users)",
				},
				"id": map[string]interface{}{
					"type":        "string",
					"description": "User ID (required for get_user)",
				},
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query (required for search_users)",
				},
			},
			"required": []string{"action"},
		},
		Handler: databaseTool,
	}

	// File system tool
	fsTool := alith.Tool{
		Name:        "file_system",
		Description: "Perform file system operations (list_files, file_info)",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action": map[string]interface{}{
					"type":        "string",
					"description": "The action to perform (list_files, file_info)",
				},
				"path": map[string]interface{}{
					"type":        "string",
					"description": "File or directory path",
				},
			},
			"required": []string{"action"},
		},
		Handler: fileSystemTool,
	}

	// Add all tools
	agent.WithTool(calcTool)
	agent.WithTool(randomTool)
	agent.WithTool(textTool)
	agent.WithTool(dbTool)
	agent.WithTool(fsTool)

	// Initialize
	if err := agent.Initialize(); err != nil {
		log.Fatal("Failed to initialize agent:", err)
	}

	// Test various prompts
	prompts := []string{
		"Calculate 15 + 27",
		"Generate a random number between 1 and 50",
		"Analyze the text: 'Hello World! This is a test.'",
		"List all users in the database",
		"Get information about user with ID 2",
		"Search for users with 'alice' in their name",
		"List files in /home/user/documents/",
		"Get information about the file /home/user/images/photo.jpg",
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

	fmt.Printf("\n✅ All multi-tool demos completed successfully!\n")
}
