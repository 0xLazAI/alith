# Contributing to Alith

We warmly welcome contributions to Alith! Whether you're a developer, a user reporting a bug, or someone eager to enhance our documentation, your participation is highly valued!

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [SDK-Specific Development](#sdk-specific-development)
  - [Python SDK](#python-sdk)
  - [Node.js SDK](#nodejs-sdk)
  - [Rust SDK](#rust-sdk)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Getting Started

Alith is a decentralized AI agent framework with support for multiple programming languages. The project consists of:

- **Core Rust Library** (`crates/`) - The main Alith framework
- **Python SDK** (`sdks/python/`) - Python bindings using PyO3
- **Node.js SDK** (`sdks/node/`) - TypeScript/JavaScript bindings using NAPI-RS
- **Rust SDK** (`sdks/rust/`) - Native Rust library

## Development Setup

### Prerequisites

- **Rust** (latest stable version) - Required for all SDKs
- **Python 3.8+** (for Python SDK)
- **Node.js 16+** (for Node.js SDK)
- **Git**

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0xLazAI/alith.git
   cd alith
   ```

2. **Install Rust dependencies:**
   ```bash
   cargo build
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file in project root
   echo "GROQ_API_KEY=your_groq_api_key" > .env
   echo "OPENAI_API_KEY=your_openai_api_key" >> .env
   ```

## SDK-Specific Development

### Python SDK

The Python SDK uses PyO3 for Rust bindings and provides a high-level Python API.

#### Setup

```bash
cd sdks/python

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install maturin for building Rust extensions
cargo install maturin

# Install development dependencies
pip install pytest ruff black
```

#### Development Commands

```bash
# Build the Rust extension
maturin develop

# Build in release mode for better performance
maturin develop --release

# Run tests
python3 -m pytest

# Lint code
ruff check

# Format code
black .

# Run examples
python3 examples/agent.py
```

#### Project Structure

```
sdks/python/
├── alith/                 # Python package
│   ├── __init__.py
│   ├── agent.py          # Main Agent class
│   ├── tool.py           # Tool definitions
│   ├── memory.py         # Memory management
│   ├── store.py          # Vector store
│   └── ...
├── examples/             # Example scripts
├── src/                  # Rust source code
│   ├── lib.rs           # PyO3 bindings
│   └── tool.rs          # Tool implementations
├── Cargo.toml           # Rust dependencies
└── pyproject.toml       # Python configuration
```

#### Key Files to Modify

- `src/lib.rs` - PyO3 bindings and Rust-Python interface
- `alith/agent.py` - Main Agent class implementation
- `alith/tool.py` - Tool system implementation
- `examples/` - Add new examples here

#### Quick Start

```bash
# 1. Navigate to Python SDK
cd sdks/python

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Build the SDK
maturin develop

# 4. Test it works
python3 examples/agent.py

# 5. Make your changes
# 6. Rebuild and test
maturin develop
python3 examples/agent.py
```

### Node.js SDK

The Node.js SDK uses NAPI-RS for Rust bindings and provides TypeScript/JavaScript support.

#### Setup

```bash
cd sdks/node

# Install dependencies
npm install
# Or use pnpm (recommended)
pnpm install

# Install additional dev dependencies
npm install --save-dev @types/json-schema
```

#### Development Commands

```bash
# Build for development (debug mode)
pnpm run build:debug

# Build for production (release mode)
pnpm run build

# Build for Windows specifically
pnpm run build:windows

# Run tests
pnpm test

# Format code
pnpm run format

# Lint code
pnpm run lint

# Fix linting issues
pnpm run lint:fix

# TypeScript compilation
pnpm run tsc

# Run examples
npx ts-node examples/agent.ts
```

#### Project Structure

```
sdks/node/
├── src/                  # TypeScript and Rust source
│   ├── agent.ts         # Main Agent class
│   ├── tool.ts          # Tool definitions
│   ├── memory.ts        # Memory management
│   ├── lib.rs           # NAPI-RS bindings
│   └── tool.rs          # Tool implementations
├── examples/             # Example scripts
├── __test__/            # Test files
├── Cargo.toml           # Rust dependencies
└── package.json         # Node.js configuration
```

#### Key Files to Modify

- `src/lib.rs` - NAPI-RS bindings and Rust-Node.js interface
- `src/agent.ts` - Main Agent class implementation
- `src/tool.ts` - Tool system implementation
- `examples/` - Add new examples here

#### Quick Start

```bash
# 1. Navigate to Node SDK
cd sdks/node

# 2. Install dependencies
pnpm install

# 3. Build the SDK
pnpm run build:debug

# 4. Test it works
npx ts-node examples/agent.ts

# 5. Make your changes
# 6. Rebuild and test
pnpm run build:debug
npx ts-node examples/agent.ts
```

#### Common Issues

**Issue: `Cannot find module '@lazai-labs/alith-win32-x64-msvc'`**

This is a common issue on Windows where the native module isn't found. Here's the complete solution:

**Step 1: Build the native module**
```bash
# Build the project
pnpm run build
```

**Step 2: Check if the native module was created**
```bash
# Look for .node files in the current directory
ls *.node
# On Windows:
dir *.node
```

**Step 3: If no .node files found, check the main project target directory**
```bash
# The native module might be in the main project's target directory
ls ../../target/release/*.dll
# On Windows:
dir ..\..\target\release\*.dll
```

**Step 4: Copy and rename the native module**
```bash
# Copy the DLL from the main project target directory
cp ../../target/release/alith_node_sdk.dll dist/

# Rename it to match the expected name
mv dist/alith_node_sdk.dll dist/alith.win32-x64-msvc.node
# On Windows:
copy ..\..\target\release\alith_node_sdk.dll dist\
ren dist\alith_node_sdk.dll alith.win32-x64-msvc.node
```

**Step 5: Alternative - Manual build process**
```bash
# If the above doesn't work, try building from the main project root
cd ../../
cargo build --release -p alith-node-sdk
cd sdks/node

# Copy the built file
cp ../../target/release/alith_node_sdk.dll dist/alith.win32-x64-msvc.node
```

**Step 6: Verify the fix**
```bash
# Check that the file exists with the correct name
ls dist/alith.win32-x64-msvc.node
# On Windows:
dir dist\alith.win32-x64-msvc.node

# Now try running your example
npx ts-node examples/agent.ts
```

**Why this happens:**
- The NAPI-RS build process creates the native module in the main project's `target/release/` directory
- The JavaScript code expects it to be named `alith.win32-x64-msvc.node` in the `dist/` directory
- The build process doesn't automatically copy and rename the file correctly on Windows

### Rust SDK

The Rust SDK is the core library that other SDKs build upon.

#### Setup

```bash
# The Rust SDK is part of the main crate
cd crates/alith

# Build the library
cargo build

# Run examples
cargo run --example agent
```

#### Development Commands

```bash
# Build the library
cargo build

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy

# Run examples
cargo run --example agent
cargo run --example agent_with_tools

# Run examples with features
cargo run --example agent_with_qdrant --features qdrant
```

#### Project Structure

```
crates/alith/
├── src/
│   └── lib.rs           # Main library exports
├── examples/            # Example programs
│   ├── agent.rs
│   ├── agent_with_tools.rs
│   └── ...
└── Cargo.toml          # Dependencies

Other crates:
├── crates/core/        # Core functionality
├── crates/client/      # LLM clients
├── crates/tools/       # Tool system
├── crates/store/       # Vector stores
└── ...
```

#### Key Files to Modify

- `crates/alith/src/lib.rs` - Main library exports
- `crates/core/src/` - Core functionality (agent, chat, llm, etc.)
- `crates/client/src/` - LLM client implementations
- `crates/tools/src/` - Tool system
- `crates/alith/examples/` - Add new examples here

#### Quick Start

```bash
# 1. Navigate to the project root
cd alith

# 2. Build the project
cargo build

# 3. Test it works
cargo run --example agent

# 4. Make your changes to crates/
# 5. Rebuild and test
cargo build
cargo run --example agent
```

#### Working with Features

```bash
# Build with specific features
cargo build --features qdrant,pgvector

# Run examples with features
cargo run --example agent_with_qdrant --features qdrant

# List all available features
cargo tree --features
```

## Contributing Guidelines

### Code Style

#### Rust
- Follow Rust conventions and use `cargo fmt`
- Use `cargo clippy` for linting
- Document public APIs with `///` comments
- Use `async_trait` for async traits
- Use `anyhow` for error handling in examples

#### Python
- Follow PEP 8 style guide
- Use `black` for formatting
- Use `ruff` for linting
- Document functions with docstrings (Google style)

#### TypeScript/JavaScript
- Use Prettier for formatting
- Follow ESLint rules (Biome for this project)
- Use JSDoc for documentation
- Follow TypeScript best practices

### Testing

#### Running Tests

```bash
# Rust tests
cargo test

# Python tests
cd sdks/python
python3 -m pytest

# Node.js tests
cd sdks/node
pnpm test
```

#### Writing Tests

**Rust:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_agent() -> Result<(), anyhow::Error> {
        let model = LLM::from_model_name("gpt-4")?;
        let agent = Agent::new("test", model);
        assert_eq!(agent.name(), "test");
        Ok(())
    }
}
```

**Python:**
```python
import pytest
from alith import Agent

def test_agent_creation():
    agent = Agent(model="gpt-4", preamble="Test")
    assert agent.model == "gpt-4"
```

**TypeScript:**
```typescript
import test from "ava";
import { Agent } from "../dist/index.js";

test("agent creation", (t) => {
  const agent = new Agent({ model: "gpt-4" });
  t.is(agent.model(), "gpt-4");
});
```

### Documentation

#### Code Documentation

- **Rust**: Use `///` for public APIs
- **Python**: Use docstrings following Google style
- **TypeScript**: Use JSDoc comments

#### README Files

Each SDK should have a comprehensive README with:
- Installation instructions
- Quick start examples
- documentation
- Development setup
- Contributing guidelines

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Test with the latest version**

### Creating a Good Issue

Include:
- **Clear title** describing the problem
- **Detailed description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, SDK version, language version, etc.)
- **Code examples** if applicable
- **Error messages** and stack traces

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `python` - Python SDK specific
- `node` - Node.js SDK specific
- `rust` - Rust SDK specific

## Pull Request Process

### Before Submitting

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run all tests** to ensure nothing breaks
7. **Format and lint** your code

### PR Guidelines

- **Clear title** describing the changes
- **Detailed description** of what was changed and why
- **Reference issues** using `#issue-number`
- **Keep PRs focused** - one feature/fix per PR
- **Update documentation** if needed
- **Add examples** if adding new features

### PR Template (Optional but Good Practice)

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All SDKs tested (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Commit Messages (Optional but Good Practice)

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Examples:
```
feat(python): add new embeddings model support
fix(node): resolve native module loading issue
docs(rust): update agent API documentation
```

## Community

### Getting Help

- **GitHub Discussions** - For questions and discussions
- **Telegram** - Join our [Telegram](https://t.me/alithai)
- **X/Twitter** - Follow [@0xalith](https://x.com/0xalith)
- **Documentation** - Check [docs](https://alith.lazai.network/docs)
- **Examples** - Look at existing examples in each SDK

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please:

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions
- **Be professional** in all interactions

## Development Workflow

### Branch Strategy

- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical fixes

### Release Process

1. **Version bump** in appropriate files
2. **Update CHANGELOG.md**
3. **Create release tag**
4. **Publish to package registries**
5. **Update documentation**

## Quick Reference

### Python SDK

```bash
cd sdks/python
python3 -m venv venv && source venv/bin/activate
maturin develop
python3 examples/agent.py
```

### Node.js SDK

```bash
cd sdks/node
pnpm install
pnpm run build:debug
npx ts-node examples/agent.ts
```

### Rust SDK

```bash
cd alith
cargo build
cargo run --example agent
```

## Troubleshooting

### Common Issues

**Python: ModuleNotFoundError**
```bash
# Make sure you've built the extension
maturin develop
```

**Node.js: Cannot find module '@lazai-labs/alith-win32-x64-msvc'**

This is a Windows-specific issue. The complete solution:

```bash
# 1. Build the project
pnpm run build

# 2. Check if .node files exist in current directory
dir *.node

# 3. If not found, check main project target directory
dir ..\..\target\release\*.dll

# 4. Copy and rename the DLL
copy ..\..\target\release\alith_node_sdk.dll dist\
ren dist\alith_node_sdk.dll alith.win32-x64-msvc.node

# 5. Verify the file exists
dir dist\alith.win32-x64-msvc.node

# 6. Now run your example
npx ts-node examples/agent.ts
```

**Alternative approach:**
```bash
# Build from main project root
cd ../../
cargo build --release -p alith-node-sdk
cd sdks/node

# Copy and rename
copy ..\..\target\release\alith_node_sdk.dll dist\alith.win32-x64-msvc.node
```

**Rust: Compilation errors**
```bash
# Update Rust and clean build
rustup update
cargo clean
cargo build
```

## Getting Started as a Contributor

1. **Start small** - Look for `good first issue` labels
2. **Read the code** - Understand the codebase structure
3. **Ask questions** - Don't hesitate to ask for help
4. **Be patient** - Learning a new codebase takes time
5. **Contribute regularly** - Even small contributions matter

## Resources

- **Website**: https://lazai.network/alith
- **Documentation**: https://alith.lazai.network/docs
- **GitHub**: https://github.com/0xLazAI/alith
- **X/Twitter**: https://x.com/0xalith
- **Telegram**: https://t.me/alithai

Thank you for contributing to Alith! 🚀

