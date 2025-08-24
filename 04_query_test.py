import chromadb
from chromadb.utils import embedding_functions

# Connect to your persistent DB
client = chromadb.PersistentClient(path="vectorstore/chroma")

# Use the same embedding model as before
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load collection
coll = client.get_collection("delhi_laws", embedding_function=embed_fn)

# Example query
query = "What are the powers of the Delhi Commission for Women?"
results = coll.query(query_texts=[query], n_results=3)

print(f"\nQuery: {query}")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
    print(f"\nResult {i}:")
    print("Source:", meta["source_path"])
    print("Chunk index:", meta["chunk_index"])
    print("Text:", doc[:500], "...")
