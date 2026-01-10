from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import Agent
import gradio as gr
from tools.orders_management import create_order, get_order_status, cancel_order
from tools.faq_tool import search_faq

# Load environment variables from .env file
load_dotenv()

# Global message history for conversation continuity
message_history = []


class SupportResponse(BaseModel):
    """Customer support agent's response"""
    response: str = Field(description="Główna treść odpowiedzi dla klienta")
    order_id: str = Field(default="", description="ID zamówienia, jeśli dotyczy")
    order_status: str = Field(default="", description="Status zamówienia, jeśli dotyczy")


SYSTEM_INSTRUCTIONS = """
Jesteś profesjonalnym agentem obsługi klienta firmy produkującej roboty przemysłowe.

TWOJE ZADANIA:
1. Pomagaj klientom w zarządzaniu zamówieniami (tworzenie, sprawdzanie statusu, anulowanie)
2. Odpowiadaj na pytania techniczne dotyczące robotów przemysłowych korzystając z bazy FAQ
3. Bądź uprzejmy, profesjonalny i pomocny

ZASADY KORZYSTANIA Z NARZĘDZI:

Zarządzanie zamówieniami:
- Aby UTWORZYĆ zamówienie użyj narzędzia create_order_tool (wymagane: nazwa użytkownika, produkt, ilość)
- Aby SPRAWDZIĆ STATUS użyj narzędzia check_order_status_tool (wymagane: ID zamówienia, nazwa użytkownika)
- Aby ANULOWAĆ zamówienie użyj narzędzia cancel_order_tool (wymagane: ID zamówienia, nazwa użytkownika, powód)

Pytania techniczne o roboty:
- Gdy klient pyta o problemy techniczne, kalibrację, przegrzewanie, błędy - ZAWSZE użyj search_faq_tool
- Bazuj odpowiedź na wynikach z FAQ, ale sformułuj ją naturalnie

WAŻNE:
- Jeśli brakuje informacji (np. nazwa użytkownika, ID zamówienia), grzecznie poproś o uzupełnienie
- Zawsze potwierdzaj wykonane akcje
- Przy problemach technicznych podawaj konkretne instrukcje z FAQ
"""

support_agent = Agent(
    'openrouter:google/gemini-2.5-flash',
    instructions=SYSTEM_INSTRUCTIONS,
    output_type=SupportResponse,
)


# === ORDER MANAGEMENT TOOLS ===

@support_agent.tool_plain
def create_order_tool(username: str, item: str, quantity: int) -> str:
    """
    Utwórz nowe zamówienie i zapisz je w bazie danych.
    
    Args:
        username: Nazwa użytkownika składającego zamówienie
        item: Nazwa produktu do zamówienia
        quantity: Ilość produktu
    
    Returns:
        Komunikat potwierdzający utworzenie zamówienia z ID
    """
    result = create_order(username, item, quantity)
    print(f"[create_order_tool] {result}")
    return result


@support_agent.tool_plain
def check_order_status_tool(order_id: str, username: str) -> str:
    """
    Sprawdź status zamówienia w bazie danych.
    
    Args:
        order_id: Identyfikator zamówienia (8-znakowy kod)
        username: Nazwa użytkownika właściciela zamówienia
    
    Returns:
        Aktualny status zamówienia
    """
    result = get_order_status(order_id, username)
    print(f"[check_order_status_tool] {result}")
    return result


@support_agent.tool_plain
def cancel_order_tool(order_id: str, username: str, reason: str) -> str:
    """
    Anuluj zamówienie i zaktualizuj jego status w bazie danych.
    
    Args:
        order_id: Identyfikator zamówienia do anulowania
        username: Nazwa użytkownika właściciela zamówienia
        reason: Powód anulowania zamówienia
    
    Returns:
        Komunikat potwierdzający anulowanie
    """
    result = cancel_order(order_id, username, reason)
    print(f"[cancel_order_tool] {result}")
    return result


# === FAQ/RAG TOOL ===

@support_agent.tool_plain
def search_faq_tool(query: str) -> str:
    """
    Wyszukaj w bazie FAQ informacje dotyczące robotów przemysłowych.
    Użyj tego narzędzia gdy klient pyta o:
    - Problemy techniczne z robotem
    - Kalibrację ramienia robota
    - Przegrzewanie się urządzeń
    - Błędy i komunikaty systemowe
    - Konserwację i serwis
    
    Args:
        query: Pytanie lub opis problemu klienta
    
    Returns:
        Odpowiednie wpisy z bazy FAQ z rozwiązaniami
    """
    result = search_faq(query, limit=3)
    print(f"[search_faq_tool] {result}")
    return result


# === GRADIO UI ===

def handle_chat(message: str, history: list[tuple[str, str]]) -> str:
    """Handle chat messages with conversation history."""
    global message_history

    result = support_agent.run_sync(message, message_history=message_history)
    
    # Update message history for conversation continuity
    message_history = result.all_messages()
    
    # Build response
    reply = result.output.response
    
    if result.output.order_id:
        reply += f"\n\n📦 **ID zamówienia:** {result.output.order_id}"
    if result.output.order_status:
        reply += f"\n📊 **Status:** {result.output.order_status}"
    
    return reply


def reset_conversation():
    """Reset conversation history."""
    global message_history
    message_history = []
    return None


demo = gr.ChatInterface(
    fn=handle_chat,
    title="🤖 Agent Wsparcia Klienta — Roboty Przemysłowe",
    description=(
        "Witaj! Jestem agentem obsługi klienta. Mogę pomóc Ci:\n"
        "- Zarządzać zamówieniami (tworzenie, sprawdzanie statusu, anulowanie)\n"
        "- Odpowiedzieć na pytania techniczne o robotach przemysłowych"
    ),
    examples=[
        "Chcę zamówić robota spawalniczego. Jestem użytkownikiem jan_kowalski",
        "Jaki jest status zamówienia abc12345? Jestem jan_kowalski",
        "Jak skalibrować ramię robota?",
        "Robot się przegrzewa, co mam zrobić?",
    ]
)


# === CLI INTERFACE ===

def run_agent(query: str) -> SupportResponse:
    """Run the agent with a single query (for evals and CLI)."""
    result = support_agent.run_sync(query)
    return result.output


if __name__ == "__main__":
    demo.launch()
