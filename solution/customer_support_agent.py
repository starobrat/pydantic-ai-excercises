from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
import gradio as gr
from tools.orders_management import create_order, get_order_status, cancel_order

# Load environment variables from .env file
load_dotenv()

# Globalna historia wiadomości
message_history = []


class SupportResponse(BaseModel):
    """Customer support agent's response"""
    response: str
    order_id: str
    order_status: str


support_agent = Agent   (
    'openrouter:google/gemini-2.5-flash',
    instructions=(
        "Jesteś specjalistą od obsługi klienta."
        "Zadbaj o to, aby klient był zadowolony z odpowiedzi."

    ),
    output_type=SupportResponse,
)


@support_agent.tool_plain
def create_order_tool(username: str, item: str, quantity: int) -> str:
    """Create a new order and save it to database"""
    return create_order(username, item, quantity)


@support_agent.tool_plain
def check_order_status_tool(order_id: str, username: str) -> str:
    """Check order status from database"""
    return get_order_status(order_id, username)


@support_agent.tool_plain
def cancel_order_tool(order_id: str, username: str, reason: str) -> str:
    """Cancel an order and update its status in database"""
    return cancel_order(order_id, username, reason)


def handle_chat(message: str, history: list[tuple[str, str]]) -> str:
    global message_history

    result = support_agent.run_sync(message, message_history=message_history)
    
    message_history = result.all_messages()
    reply = result.output.response
    
    if result.output.order_id:
        reply += f"\n\nID zamówienia: {result.output.order_id}"
    if result.output.order_status:
        reply += f"\nStatus zamówienia: {result.output.order_status}"
    
    return reply


demo = gr.ChatInterface(
    fn=handle_chat,
    title="Pydantic AI — Agent wsparcia klienta",
    description=(
        "Agent obsługi klienta pomagający w zarządzaniu zamówieniami."
    ),
)


if __name__ == "__main__":
    demo.launch()



def main_cli():
    """Simple example of running the customer support agent"""
    
    query = "Chcę kupić laptopa"
    # query = "Jaki jest status zamówienia d12ccb06? Jestem piotrstarobrat"
    # query = "Anuluj zamówienie d12ccb06. Jestem piotrstarobrat"

    result = support_agent.run_sync(query)
    print(f"Customer Query: {query}")
    print(f"Support Response: {result.output.response}")