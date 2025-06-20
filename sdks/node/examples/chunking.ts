import { chunkText } from "alith";

console.log(
  chunkText(`
## Introduction

Alith is an AI agent framework designed for the Web3 and Crypto, empowering developers to quickly build, deploy, and manage on-chain AI agents. By integrating blockchain technology, AI/ML models, and developer-friendly tools, Alith provides a modular and extensible platform that supports AI bots, multi-chain interactions, smart contract operations, real-time data processing, and high-performance inference. Whether you're building DeFi trading bots, NFT analyzers, or DAO governance tools, Alith offers robust support for your projects. For more indroduction and comparision, see documents [here](https://alith.lazai.network/docs).

## Features

+ **Multiple Model Support**: Support for small models and large language models e.g., Llama, Grok, OpenAI, Anthropic, etc.
+ **Highly Extensible**: From covering internal prompts to accessing low-level APIs. Customize roles, goals, tools, actions and behaviors while maintaining a clean abstraction.
+ **Workflow Support**: Implementing any workflow pattern through processes - from simple sequential and hierarchical processes to complex custom orchestration patterns with conditional branching and parallel execution.
+ **Cross-Language Support**: Provides SDKs for Rust, Python and Node.js, making it easily accessible to different developers.
+ **High-Performance Inference**: Leverage the performance advantages of Rust and quick model inference technologies including graph optimization, model compression, JIT/AOT compilation with GPU coprocessors, etc.
+ **Web3 Friendly and Secure**: Out-of-the-box Web3 plugin allows developers to easily and securely integrate blockchain capabilities into TEE-based AI agent frameworks and their applications.
    `)
);
