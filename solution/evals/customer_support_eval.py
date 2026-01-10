"""
Evaluations for Customer Support Agent.

This module tests the agent's ability to:
1. Handle order management requests
2. Answer technical FAQ questions about robots
3. Provide helpful and professional responses
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance
from dotenv import load_dotenv
from customer_support_agent import run_agent, SupportResponse

load_dotenv()


# === TEST CASES ===

customer_support_dataset = Dataset(
    cases=[
        # Order Management Cases
        Case(
            name='status_zamowienia',
            inputs='Gdzie jest moja paczka? Zamówienie abc12345, użytkownik jan_kowalski',
            metadata={'category': 'order_status'},
        ),
        Case(
            name='anulowanie_zamowienia',
            inputs='Chcę anulować zamówienie xyz98765. Jestem piotr_nowak. Powód: zmiana decyzji',
            metadata={'category': 'order_cancel'},
        ),
        Case(
            name='utworzenie_zamowienia',
            inputs='Chcę zamówić 2 roboty spawalnicze. Jestem użytkownikiem anna_wisniewska',
            metadata={'category': 'order_create'},
        ),
        
        # FAQ/RAG Cases - Technical Questions
        Case(
            name='faq_kalibracja',
            inputs='Jak skalibrować ramię robota?',
            metadata={'category': 'faq_technical'},
        ),
        Case(
            name='faq_przegrzewanie',
            inputs='Robot się przegrzewa podczas pracy. Co mam zrobić?',
            metadata={'category': 'faq_technical'},
        ),
        Case(
            name='faq_blad_systemu',
            inputs='Wyświetla się błąd E-101 na panelu robota',
            metadata={'category': 'faq_technical'},
        ),
        
        # Mixed/General Cases
        Case(
            name='brak_danych',
            inputs='Chcę sprawdzić status zamówienia',
            metadata={'category': 'incomplete_request'},
        ),
    ],
    evaluators=[
        # Basic type check
        IsInstance(type_name='SupportResponse'),
        
        # Professional tone evaluation
        LLMJudge(
            rubric='''
            Oceń czy odpowiedź jest uprzejma i profesjonalna.
            Odpowiedź powinna:
            1. Być napisana w grzecznym, pomocnym tonie
            2. Nie zawierać niegrzecznych lub lekceważących sformułowań
            3. Traktować klienta z szacunkiem
            ''',
            include_input=True,
            assertion={'evaluation_name': 'professional_tone'},
        ),
        
        # Helpfulness evaluation
        LLMJudge(
            rubric='''
            Oceń czy odpowiedź jest pomocna i zawiera konkretne informacje.
            Odpowiedź powinna:
            1. Bezpośrednio odpowiadać na pytanie klienta
            2. Zawierać konkretne instrukcje lub informacje
            3. Jeśli brakuje danych - prosić o ich uzupełnienie
            4. Nie być ogólnikowa ani wymijająca
            ''',
            include_input=True,
            score={'evaluation_name': 'helpfulness'},
            assertion=False,
        ),
        
        # Completeness evaluation
        LLMJudge(
            rubric='''
            Oceń kompletność odpowiedzi.
            Dla pytań o zamówienia:
            - Odpowiedź powinna zawierać informacje o statusie lub potwierdzenie akcji
            - Jeśli brakuje danych, powinna jasno wskazać jakich
            
            Dla pytań technicznych o roboty:
            - Odpowiedź powinna zawierać konkretne instrukcje lub rozwiązania
            - Powinna być oparta na wiedzy technicznej (z FAQ)
            ''',
            include_input=True,
            score={'evaluation_name': 'completeness'},
            assertion=False,
        ),
    ],
)


# === ADDITIONAL DETAILED EVALUATION ===

detailed_faq_dataset = Dataset(
    cases=[
        Case(
            name='faq_kalibracja_detailed',
            inputs='Jak skalibrować ramię robota przemysłowego?',
        ),
        Case(
            name='faq_serwis',
            inputs='Kiedy powinienem przeprowadzić serwis robota?',
        ),
        Case(
            name='faq_bezpieczenstwo',
            inputs='Jakie są zasady bezpieczeństwa przy pracy z robotem?',
        ),
    ],
    evaluators=[
        IsInstance(type_name='SupportResponse'),
        
        # Technical accuracy
        LLMJudge(
            rubric='''
            Odpowiedź na pytanie techniczne powinna:
            1. Zawierać konkretne kroki lub instrukcje
            2. Być technicznie sensowna (nawet jeśli ogólna)
            3. Nie zawierać błędnych lub niebezpiecznych porad
            ''',
            include_input=True,
            assertion={'evaluation_name': 'technical_accuracy'},
        ),
        
        # Actionability score
        LLMJudge(
            rubric='''
            Oceń czy klient może podjąć konkretne działania na podstawie odpowiedzi.
            Wysoka ocena: odpowiedź zawiera jasne, wykonalne kroki
            Niska ocena: odpowiedź jest ogólnikowa, bez konkretnych instrukcji
            ''',
            include_input=True,
            score={'evaluation_name': 'actionability'},
            assertion=False,
        ),
    ],
)


def run_evaluation():
    """Run the main evaluation dataset."""
    print("=" * 60)
    print("🔍 EWALUACJA AGENTA OBSŁUGI KLIENTA")
    print("=" * 60)
    print("\n📋 Główny dataset ewaluacyjny\n")
    
    report = customer_support_dataset.evaluate_sync(run_agent)
    
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
    )
    
    return report


def run_detailed_faq_evaluation():
    """Run detailed FAQ evaluation."""
    print("\n" + "=" * 60)
    print("🔬 SZCZEGÓŁOWA EWALUACJA FAQ")
    print("=" * 60 + "\n")
    
    report = detailed_faq_dataset.evaluate_sync(run_agent)
    
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
    )
    
    return report


def main():
    """Run all evaluations."""
    print("\n🚀 Rozpoczynam ewaluację agenta obsługi klienta...\n")
    
    # Main evaluation
    main_report = run_evaluation()
    
    # Detailed FAQ evaluation
    faq_report = run_detailed_faq_evaluation()
    
    print("\n" + "=" * 60)
    print("✅ EWALUACJA ZAKOŃCZONA")
    print("=" * 60)


if __name__ == "__main__":
    main()
