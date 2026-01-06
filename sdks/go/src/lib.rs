// sdks/go/src/lib.rs
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use alith::{Agent, Chat, LLM, Tool, ToolDefinition, ToolError};
use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;
use async_trait::async_trait;
use serde_json;

// Global storage
lazy_static! {
    static ref AGENTS: Mutex<HashMap<String, Agent<LLM>>> = Mutex::new(HashMap::new());
    static ref TOOLS: Mutex<HashMap<u64, GoTool>> = Mutex::new(HashMap::new());
    static ref COUNTER: Mutex<usize> = Mutex::new(0);
}

// External Go callback function
unsafe extern "C" {
    fn go_tool_callback(tool_id: u64, input: *const c_char) -> *mut c_char;
}

// Tool that calls back to Go
#[derive(Clone)]
struct GoTool {
    name: String,
    description: String,
    parameters: String,
    tool_id: u64,
}

#[async_trait]
impl Tool for GoTool {
    fn name(&self) -> &str { &self.name }
    fn description(&self) -> &str { &self.description }
    fn version(&self) -> &str { "1.0.0" }
    fn author(&self) -> &str { "Go" }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: serde_json::from_str(&self.parameters).unwrap_or_default(),
        }
    }
    
    async fn run(&self, input: &str) -> Result<String, ToolError> {
        unsafe {
            let c_input = CString::new(input).map_err(|_| ToolError::InvalidInput)?;
            let c_result = go_tool_callback(self.tool_id, c_input.as_ptr());
            
            if c_result.is_null() {
                return Err(ToolError::InvalidOutput);
            }
            
            let result = CStr::from_ptr(c_result).to_string_lossy().into_owned();
            libc::free(c_result as *mut libc::c_void); // Free Go-allocated memory
            Ok(result)
        }
    }
}

// Create agent
#[unsafe(no_mangle)]
pub extern "C" fn create_agent(
    name: *const c_char,
    model: *const c_char,
    api_key: *const c_char,
    base_url: *const c_char,
    preamble: *const c_char
) -> *mut c_char {
    let name_str = unsafe { CStr::from_ptr(name).to_string_lossy() };
    let model_str = unsafe { CStr::from_ptr(model).to_string_lossy() };
    let api_key_str = unsafe { CStr::from_ptr(api_key).to_string_lossy() };
    let base_url_str = unsafe { CStr::from_ptr(base_url).to_string_lossy() };
    let preamble_str = unsafe { CStr::from_ptr(preamble).to_string_lossy() };
    
    let llm = if base_url_str.is_empty() {
        match LLM::from_model_name(&model_str) {
            Ok(llm) => llm,
            Err(e) => return CString::new(format!("ERROR: {}", e)).unwrap().into_raw(),
        }
    } else {
        match LLM::openai_compatible_model(&api_key_str, &base_url_str, &model_str) {
            Ok(llm) => llm,
            Err(e) => return CString::new(format!("ERROR: {}", e)).unwrap().into_raw(),
        }
    };
    
    let mut agent = Agent::new(&name_str, llm);
    if !preamble_str.is_empty() {
        agent = agent.preamble(&preamble_str);
    }
    
    let mut agents = AGENTS.lock().unwrap();
    let mut counter = COUNTER.lock().unwrap();
    *counter += 1;
    let agent_id = format!("agent_{}", *counter);
    agents.insert(agent_id.clone(), agent);
    
    CString::new(agent_id).unwrap().into_raw()
}

// Add tool to agent
#[unsafe(no_mangle)]
pub extern "C" fn add_tool_to_agent(
    agent_id: *const c_char,
    tool_name: *const c_char,
    tool_description: *const c_char,
    tool_parameters: *const c_char,
    func_ptr: u64
) -> *mut c_char {
    let agent_id_str = unsafe { CStr::from_ptr(agent_id).to_string_lossy() };
    let name = unsafe { CStr::from_ptr(tool_name).to_string_lossy() };
    let description = unsafe { CStr::from_ptr(tool_description).to_string_lossy() };
    let parameters = unsafe { CStr::from_ptr(tool_parameters).to_string_lossy() };
    
    let tool = GoTool {
        name: name.to_string(),
        description: description.to_string(),
        parameters: parameters.to_string(),
        tool_id: func_ptr,
    };
    
    let mut agents = AGENTS.lock().unwrap();
    let agent_id_owned = agent_id_str.to_string();
    if let Some(agent) = agents.remove(&agent_id_owned) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let updated_agent = rt.block_on(agent.tool(tool));
        agents.insert(agent_id_owned, updated_agent);
        CString::new("OK").unwrap().into_raw()
    } else {
        CString::new("ERROR: Agent not found").unwrap().into_raw()
    }
}

// Prompt agent
#[unsafe(no_mangle)]
pub extern "C" fn agent_prompt(
    agent_id: *const c_char,
    prompt: *const c_char
) -> *mut c_char {
    let agent_id_str = unsafe { CStr::from_ptr(agent_id).to_string_lossy() };
    let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy() };
    
    let mut agents = AGENTS.lock().unwrap();
    if let Some(agent) = agents.get_mut(&agent_id_str.to_string()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(agent.prompt(&prompt_str)) {
            Ok(response) => CString::new(response).unwrap().into_raw(),
            Err(e) => CString::new(format!("ERROR: {}", e)).unwrap().into_raw(),
        }
    } else {
        CString::new("ERROR: Agent not found").unwrap().into_raw()
    }
}

// Free memory allocated by Rust
#[unsafe(no_mangle)]
pub extern "C" fn free_rust_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { CString::from_raw(ptr) };
    }
}