import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize sentence transformer model for embeddings
# Using multilingual model for cross-language search (supports Polish, English, etc.)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def calculate_similarity(query1: str, query2: str) -> None:
    """
    Calculate and display semantic similarity between two sentences.
    
    Args:
        query1: First sentence
        query2: Second sentence
    """
    # Generate embeddings for both queries
    embedding1 = model.encode(query1)
    embedding2 = model.encode(query2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Display results
    print(f"Query 1: {query1}")
    print(f"Query 2: {query2}\n")
    
    print(f"Embedding 1 preview (first 20 numbers):")
    print(f"{embedding1[:20]}\n")
    
    print(f"Embedding 2 preview (first 20 numbers):")
    print(f"{embedding2[:20]}\n")
    
    print(f"Semantic Similarity (cosine similarity): {similarity:.4f}")
    print(f"Similarity percentage: {similarity * 100:.2f}%")


def main():
    """Main function to execute semantic similarity comparison."""
    query1 = "Dowcip o kotach"
    query2 = "Zabawna zagadka o Mruczku"
    
    try:
        calculate_similarity(query1, query2)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
