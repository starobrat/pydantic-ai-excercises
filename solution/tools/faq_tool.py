import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize Qdrant client
_qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Collection name for robot support FAQ
COLLECTION_NAME = "customer-service-robot-support"

# Initialize sentence transformer model for embeddings (eager loading)
_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def search_faq(query: str, limit: int = 3) -> str:
    """
    Search FAQ database for relevant answers using semantic search.
    
    Args:
        query: User's question or problem description
        limit: Number of results to return (default: 3)
    
    Returns:
        Formatted string with relevant FAQ entries
    """
    
    # Generate embedding for the query
    query_embedding = _model.encode(query).tolist()
    
    # Perform vector search
    search_results = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
    ).points
    
    if not search_results:
        return "Nie znaleziono odpowiednich informacji w bazie FAQ."
    
    # Format results
    formatted_results = []
    for i, result in enumerate(search_results, 1):
        payload = result.payload
        print(payload)
        description = payload.get("description", "Brak opisu")
        dialogue = payload.get("dialogue", "Brak dialogu")
        score = result.score
        
        formatted_results.append(
            f"--- Wynik {i} (trafność: {score:.2f}) ---\n"
            f"Opis problemu: {description}\n"
            f"Rozwiązanie:\n{dialogue}"
        )
    
    return "\n\n".join(formatted_results)
