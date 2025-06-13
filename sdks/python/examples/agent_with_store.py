from pathlib import Path

from alith import Agent, MilvusStore, chunk_text

print(
    Agent(
        name="RAG Bot",
        model="gpt-4",
        preamble="I'm a RAG bot. Ask me anything!",
        store=MilvusStore().save_docs(chunk_text(Path("README.md").read_text())),
    ).prompt("What is Alith?")
)
