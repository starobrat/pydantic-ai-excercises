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

def main():
    result = agent.run_sync("Opowiedz mi dowcip o kotach.")
    print(f"Joke: {result.output.joke}")
    print(f"Joke Type: {result.output.joke_type}")
    print(f"All messages: {result.all_messages}")


if __name__ == "__main__":
    main()