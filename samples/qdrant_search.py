import os
import sys
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Get collection name
collection_name = "customer-service-robot-support"

# Initialize sentence transformer model for embeddings
# Using multilingual model for cross-language search (supports Polish, English, etc.)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def search_query(query: str, limit: int = 5) -> list:
    """
    Perform vector search for a given query.
    
    Args:
        query: The search query text
        limit: Number of results to return (default: 5)
    
    Returns:
        List of search results with scores
    """
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()
    
    # Perform vector search using query_points
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
    ).points
    
    return search_results


def display_results(query: str, results: list):
    """Display search results in a readable format."""
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        payload = result.payload
        print(f"\nResult {i} (Score: {result.score:.4f})")
        print(f"ID: {result.id} (original: {payload.get('original_id', 'N/A')})")
        print(f"Description: {payload.get('description', 'N/A')}")
        print(f"Dialogue:")
        print(f"  {payload.get('dialogue', 'N/A')}")
        print("-" * 80)


def main():
    """Main function to execute vector search."""
    query = "Przegrzewa się robot, co zrobić?"
    limit = 5

    try:
        # Perform search
        results = search_query(query, limit=limit)
        
        # Display results
        display_results(query, results)
        
    except Exception as e:
        print(f"Error performing search: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
