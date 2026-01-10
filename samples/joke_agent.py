from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
import random

# Load environment variables from .env file
load_dotenv()


class JokeResponse(BaseModel):
    joke: str
    joke_type: str


agent = Agent   (
    'openrouter:anthropic/claude-haiku-4.5',
    instructions=(
        "Jesteś specjalistą od opowiadania dowcipów."
        "Wylosuj typ dowcipu i opowiedz dowcip w wylosowanym typie."
    ),
    output_type=JokeResponse,
)

@agent.tool_plain
def get_joke_topic_tool() -> str:
    joke_types = ["suchar", "czarny humor", "zagadka"]
    selected_type = random.choice(joke_types)
    return selected_type


def run_joke_agent(topic: str) -> JokeResponse:
    """Run the joke agent with a given topic. Used by pydantic-evals."""
    result = agent.run_sync(f"Opowiedz mi dowcip o {topic}.")
    return result.output


def main():
    output = run_joke_agent("kotach")
    print(f"Joke: {output.joke}")
    print(f"Joke Type: {output.joke_type}")


if __name__ == "__main__":
    main()