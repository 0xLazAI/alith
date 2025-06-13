from dataclasses import dataclass

from pydantic import BaseModel

from .agent import Agent
from .tool import Tool


@dataclass
class Extractor:
    """Structure data extractor based on the agent"""

    agent: Agent
    model: type[BaseModel]

    def extract(self, input: str) -> BaseModel:
        """Extract structure data from an input string."""
        agent = Agent(
            name=self.agent.name,
            model=self.agent.model,
            preamble="""Extract the data structure from the input string.
Note you MUST use the tool named `extractor` to extract the input string to the
data structure.
""",
            api_key=self.agent.api_key,
            base_url=self.agent.base_url,
            tools=[
                Tool(
                    name="extractor",
                    description="Extract the data structure from the input string.",
                    parameters=self.model,
                    handler=lambda **kwargs: kwargs,
                )
            ],
        )
        result = agent.prompt(input)
        return self.model.model_validate_json(result)
