from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
import random
import gradio as gr

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()


class JokeResponse(BaseModel):
    joke: str
    joke_type: str


agent = Agent(
    "openrouter:anthropic/claude-haiku-4.5",
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


def handle_chat(message: str, history: list[tuple[str, str]]) -> str:
    result = agent.run_sync(message)
    reply = f"{result.output.joke}\n\nTyp dowcipu: {result.output.joke_type}"
    return reply


demo = gr.ChatInterface(
    fn=handle_chat,
    title="Pydantic AI — Agent od opowiadania dowcipów",
    description=(
        "Agent opowiadający dowcipy."
    ),
)


if __name__ == "__main__":
    demo.launch()

