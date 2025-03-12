from alith import Agent, Extractor
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


print(
    Extractor(
        Agent(
            model="gpt-4",
        ),
        Person,
    ).extract("Alice is 18 years old!")
)
