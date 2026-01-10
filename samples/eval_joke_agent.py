from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance
from dotenv import load_dotenv
from joke_agent import run_joke_agent

load_dotenv()


joke_dataset = Dataset(
    cases=[
        Case(
            name='dowcip_o_kotach',
            inputs='kotach',
        ),
        Case(
            name='dowcip_o_programistach',
            inputs='programistach',
        ),
        Case(
            name='dowcip_o_lekarzach',
            inputs='lekarzach',
        ),
        Case(
            name='dowcip_o_politykach',
            inputs='politykach',
        ),
        Case(
            name='dowcip_o_szkole',
            inputs='szkole',
        ),
    ],
    evaluators=[
        # Sprawdź czy output to JokeResponse
        IsInstance(type_name='JokeResponse'),
        
        # Czy dowcip jest na temat?
        LLMJudge(
            rubric='''
            Dowcip musi być związany z tematem podanym w inputs.
            Jeśli inputs to "kotach", dowcip powinien być o kotach.
            ''',
            include_input=True,
            assertion={'evaluation_name': 'on_topic'}
        ),
        
        # Czy dowcip jest śmieszny?
        LLMJudge(
            rubric='''
            Oceń czy dowcip jest zabawny i ma sens.
            Dowcip powinien mieć puentę lub element zaskoczenia.
            ''',
            include_input=True,
            score={'evaluation_name': 'humor_quality'},
            assertion=False,
        ),
        
        # Czy joke_type pasuje do treści?
        LLMJudge(
            rubric='''
            Sprawdź czy pole joke_type odpowiada faktycznemu typowi dowcipu:
            - "suchar" - prosty, przewidywalny dowcip, często gra słów
            - "czarny humor" - kontrowersyjny, mroczny temat
            - "zagadka" - format pytanie-odpowiedź
            ''',
            include_input=True,
            assertion={'evaluation_name': 'type_matches'},
        ),
    ],
)


def main():
    print("Uruchamiam ewaluację agenta od dowcipów...\n")
    
    report = joke_dataset.evaluate_sync(run_joke_agent)
    
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
    )


if __name__ == "__main__":
    main()