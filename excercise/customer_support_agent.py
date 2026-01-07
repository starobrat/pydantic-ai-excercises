from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

# Load environment variables from .env file
load_dotenv()





class SupportResponse(BaseModel):
    """Customer support agent's response"""
    response: str
    was_completed: bool


support_agent = Agent   (
    # TODO 1: go to https://openrouter.ai/models, choose model and replace XXX with the model name
    'openrouter:XXX',
    instructions=(
        # TODO 2: Add agent instruction to be a customer support agent
        "TODO: Add instructions here"
    ),
    output_type=SupportResponse,
)


def main():
    """Simple example of running the customer support agent"""
    
    # TODO 3: Add sample query to be a customer support query and replace XXX with the query
    query = "XXX"
    
    result = support_agent.run_sync(query)
    print(f"Customer Query: {query}")
    print(f"Support Response: {result.output.response}")


if __name__ == "__main__":
    main()
