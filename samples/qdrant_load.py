import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("Existing collections:", qdrant_client.get_collections())

# Load all subsets
dataset = load_dataset("FunDialogues/customer-service-robot-support")

# Get collection name
collection_name = "customer-service-robot-support"

# Initialize sentence transformer model for embeddings
# Using multilingual model for cross-language search (supports Polish, English, etc.)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vector_size = 384

# Check if collection exists, create if not
existing_collections = [col.name for col in qdrant_client.get_collections().collections]
collection_exists = collection_name in existing_collections

if not collection_exists:
    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Created collection '{collection_name}' with vector size {vector_size}")
else:
    print(f"Collection '{collection_name}' already exists")

# Check if collection is empty
collection_info = qdrant_client.get_collection(collection_name)
is_collection_empty = collection_info.points_count == 0

if not is_collection_empty:
    print(f"Collection '{collection_name}' already contains {collection_info.points_count} points. Skipping data load.")
else:
    # Load dataset into Qdrant
    train_data = dataset["train"]
    batch_size = 100
    total_points = 0
    current_batch = []

    for idx, item in enumerate(train_data):
        # Combine description and dialogue for better semantic search
        text_to_embed = f"{item['description']}\n{item['dialogue']}"
        
        # Generate embedding
        embedding = model.encode(text_to_embed).tolist()
        
        # Prepare payload (keep original id from dataset for reference)
        payload = {
            "original_id": item["id"],
            "description": item["description"],
            "dialogue": item["dialogue"],
        }
        
        # Use index as unique ID (dataset has duplicate 'id' values!)
        current_batch.append(
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload=payload,
            )
        )
        
        # Upsert when batch is full
        if len(current_batch) >= batch_size:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=current_batch,
            )
            total_points += len(current_batch)
            print(f"Upserted batch of {len(current_batch)} points (total: {total_points})")
            current_batch = []

    # Upsert remaining points
    if current_batch:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=current_batch,
        )
        total_points += len(current_batch)
        print(f"Upserted final batch of {len(current_batch)} points (total: {total_points})")

    print(f"Successfully loaded {total_points} points into collection '{collection_name}'")
