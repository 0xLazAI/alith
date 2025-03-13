# Langchain Alith Python Integration

This integration provides the following methods:

1. **Enable the Alith chain in Langchain**: We can use the Alith as the LLM node for the existing Langchain workflow and get the performance gains of Alith.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_alith import LLM
from alith import Agent

prompt = PromptTemplate.from_template(
    """As an adaptable question-answering assistant, your role is to leverage the provided context to address user inquiries. When direct answers are not apparent from the context, you are encouraged to draw upon analogies or related knowledge to formulate or infer solutions. If a certain answer remains elusive, politely acknowledge the limitation. Aim for concise responses, ideally within three sentences. In response to requests for links, explain that link provision is not supported.

Question: {question}
"""
)


def main():
    llm = LLM(
        agent=Agent(
            model="gpt-4",
            preamble="You are a comedian here to entertain the user using humour and jokes.",
        )
    )
    chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    print(chain.invoke("query"))


if __name__ == "__main__":
    main()
```
