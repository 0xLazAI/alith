pub use llm_client::utils::chunking::{chunk_text, ChunkerConfig, ChunkerResult, TextChunker};

pub const DEFAULT_CHUNK_SIZE: usize = 1024;

pub trait Chunk: Send + Sync {
    fn chunk_size(&self) -> usize {
        DEFAULT_CHUNK_SIZE
    }

    fn overlap_percent(&self) -> Option<f32> {
        None
    }

    fn chunk(&self) -> Result<Vec<String>, ChunkError>;
}

/// An enumeration of possible errors that may occur during chunk operations.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    /// A generic chunk error.
    #[error("A normal chunk error occurred: {0}")]
    Normal(String),
}

#[cfg(test)]
mod tests {
    use super::chunk_text;

    #[test]
    fn test_chunk_text() {
        let text = r#"
## Introduction

Alith is an AI agent framework designed for the Web3 and Crypto, empowering developers to quickly build, deploy, and manage on-chain AI agents. By integrating blockchain technology, AI/ML models, and developer-friendly tools, Alith provides a modular and extensible platform that supports AI bots, multi-chain interactions, smart contract operations, real-time data processing, and high-performance inference. Whether you're building DeFi trading bots, NFT analyzers, or DAO governance tools, Alith offers robust support for your projects. For more indroduction and comparision, see documents [here](https://alith.lazai.network/docs).

## Features

+ **Multiple Model Support**: Support for small models and large language models e.g., Llama, Grok, OpenAI, Anthropic, etc.
+ **Highly Extensible**: From covering internal prompts to accessing low-level APIs. Customize roles, goals, tools, actions and behaviors while maintaining a clean abstraction.
+ **Workflow Support**: Implementing any workflow pattern through processes - from simple sequential and hierarchical processes to complex custom orchestration patterns with conditional branching and parallel execution.
+ **Cross-Language Support**: Provides SDKs for Rust, Python and Node.js, making it easily accessible to different developers.
+ **High-Performance Inference**: Leverage the performance advantages of Rust and quick model inference technologies including graph optimization, model compression, JIT/AOT compilation with GPU coprocessors, etc.
+ **Web3 Friendly and Secure**: Out-of-the-box Web3 plugin allows developers to easily and securely integrate blockchain capabilities into TEE-based AI agent frameworks and their applications.
"#;
        assert_eq!(
            chunk_text(text, 200, None)
                .unwrap()
                .unwrap_or_default()
                .len(),
            2
        );
        assert_eq!(
            chunk_text(text, 40, None)
                .unwrap()
                .unwrap_or_default()
                .len(),
            9
        );
        assert_eq!(
            chunk_text(text, 40, Some(0.1))
                .unwrap()
                .unwrap_or_default()
                .len(),
            10
        );
    }
}
