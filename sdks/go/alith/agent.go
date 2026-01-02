// sdks/go/alith/agent.go
package alith

/*
#cgo LDFLAGS: -L. -l:libalith_go_sdk.a -lws2_32 -luserenv -lbcrypt -lntdll
#include <stdlib.h>

extern char* create_agent(char* name, char* model, char* api_key, char* base_url, char* preamble);
extern char* add_tool_to_agent(char* agent_id, char* tool_name, char* tool_description, char* tool_parameters, unsigned long long func_ptr);
extern char* agent_prompt(char* agent_id, char* prompt);
extern void free_rust_string(char* ptr);

// Forward declaration for Go callback
char* go_tool_callback(unsigned long long tool_id, char* input);
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"
)

var (
	toolRegistry   = make(map[uint64]func(map[string]interface{}) (interface{}, error))
	toolRegistryMu sync.RWMutex
	nextToolID     uint64 = 1
)

type Agent struct {
	ID       string
	Name     string
	Model    string
	APIKey   string
	BaseURL  string
	Preamble string
	tools    []Tool
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Handler     func(map[string]interface{}) (interface{}, error)
}

// This is called from Rust via C
//
//export go_tool_callback
func go_tool_callback(toolID C.ulonglong, input *C.char) *C.char {
	inputStr := C.GoString(input)

	toolRegistryMu.RLock()
	handler, exists := toolRegistry[uint64(toolID)]
	toolRegistryMu.RUnlock()

	if !exists {
		return C.CString(`{"error": "tool not found"}`)
	}

	var inputMap map[string]interface{}
	if err := json.Unmarshal([]byte(inputStr), &inputMap); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%s"}`, err.Error()))
	}

	result, err := handler(inputMap)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%s"}`, err.Error()))
	}

	resultBytes, _ := json.Marshal(result)
	return C.CString(string(resultBytes))
}

// Create new agent
func NewAgent(name, model string) *Agent {
	return &Agent{
		Name:  name,
		Model: model,
		tools: make([]Tool, 0),
	}
}

// Set API credentials
func (a *Agent) WithCredentials(apiKey, baseURL string) *Agent {
	a.APIKey = apiKey
	a.BaseURL = baseURL
	return a
}

// Set preamble
func (a *Agent) WithPreamble(preamble string) *Agent {
	a.Preamble = preamble
	return a
}

// Add tool
func (a *Agent) WithTool(tool Tool) *Agent {
	a.tools = append(a.tools, tool)
	return a
}

// Initialize the agent (creates Rust agent)
func (a *Agent) Initialize() error {
	cName := C.CString(a.Name)
	cModel := C.CString(a.Model)
	cAPIKey := C.CString(a.APIKey)
	cBaseURL := C.CString(a.BaseURL)
	cPreamble := C.CString(a.Preamble)

	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cModel))
	defer C.free(unsafe.Pointer(cAPIKey))
	defer C.free(unsafe.Pointer(cBaseURL))
	defer C.free(unsafe.Pointer(cPreamble))

	result := C.create_agent(cName, cModel, cAPIKey, cBaseURL, cPreamble)
	defer C.free_rust_string(result)

	resultStr := C.GoString(result)
	if len(resultStr) > 6 && resultStr[:6] == "ERROR:" {
		return fmt.Errorf(resultStr[7:])
	}

	a.ID = resultStr

	// Add tools
	for _, tool := range a.tools {
		if err := a.addToolToRust(tool); err != nil {
			return err
		}
	}

	return nil
}

// Add tool to Rust agent
func (a *Agent) addToolToRust(tool Tool) error {
	// Register the tool handler and get an ID
	toolRegistryMu.Lock()
	toolID := nextToolID
	nextToolID++
	toolRegistry[toolID] = tool.Handler
	toolRegistryMu.Unlock()

	// Convert parameters to JSON
	paramsBytes, _ := json.Marshal(tool.Parameters)

	cAgentID := C.CString(a.ID)
	cToolName := C.CString(tool.Name)
	cToolDesc := C.CString(tool.Description)
	cToolParams := C.CString(string(paramsBytes))

	defer C.free(unsafe.Pointer(cAgentID))
	defer C.free(unsafe.Pointer(cToolName))
	defer C.free(unsafe.Pointer(cToolDesc))
	defer C.free(unsafe.Pointer(cToolParams))

	result := C.add_tool_to_agent(cAgentID, cToolName, cToolDesc, cToolParams, C.ulonglong(toolID))
	defer C.free_rust_string(result)

	resultStr := C.GoString(result)
	if resultStr != "OK" {
		return fmt.Errorf("failed to add tool: %s", resultStr)
	}

	return nil
}

// Prompt the agent
func (a *Agent) Prompt(prompt string) (string, error) {
	if a.ID == "" {
		return "", fmt.Errorf("agent not initialized - call Initialize() first")
	}

	cAgentID := C.CString(a.ID)
	cPrompt := C.CString(prompt)

	defer C.free(unsafe.Pointer(cAgentID))
	defer C.free(unsafe.Pointer(cPrompt))

	result := C.agent_prompt(cAgentID, cPrompt)
	defer C.free_rust_string(result)

	resultStr := C.GoString(result)
	if len(resultStr) > 6 && resultStr[:6] == "ERROR:" {
		return "", fmt.Errorf(resultStr[7:])
	}

	return resultStr, nil
}
