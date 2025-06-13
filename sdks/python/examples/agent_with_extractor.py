from pydantic import BaseModel

from alith import Agent, Extractor


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
