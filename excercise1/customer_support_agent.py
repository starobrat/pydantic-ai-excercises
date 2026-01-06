import os
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

# Load environment variables from .env file
load_dotenv()


class SupportResponse(BaseModel):
    """Customer support agent's response"""
    response: str
    helpful: bool

# Create customer support agent
support_agent = Agent   (
    'openrouter:TODO: Add model here',
    instructions=(
        "TODO: Add instructions here"
    ),
    output_type=SupportResponse,
)


def main():
    """Simple example of running the customer support agent"""
    query = "TODO: Add query here"
    
    result = support_agent.run_sync(query)
    print(f"Customer Query: {query}")
    print(f"Support Response: {result.output.response}")


if __name__ == "__main__":
    main()

