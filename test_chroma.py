import chromadb
import ollama

# Set up a persistent database, saved to a folder called chroma_db
client = chromadb.PersistentClient(path="chroma_db")

# Create (or get) a collection - think of this like a table in a normal database
collection = client.get_or_create_collection(name="test_collection")

# Some sample chunks to store
sample_chunks = [
    "The dog ran fast across the yard.",
    "I love eating pizza on Fridays.",
    "Nanoparticles can be used to deliver drugs directly to tumor cells."
]

# Embed each chunk and add it to the collection
for i, chunk in enumerate(sample_chunks):
    embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("Chunks stored! Now let's search...")

# Now ask a question and see which chunk is most similar
question = "How can medicine be delivered to cancer cells?"
question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)

print("Most relevant chunk found:")
print(results['documents'][0][0])