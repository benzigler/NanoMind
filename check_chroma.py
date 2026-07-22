import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="papers")

print(f"Total chunks stored: {collection.count()}")

# Peek at a few entries
results = collection.peek(limit=3)
for doc_id, doc in zip(results['ids'], results['documents']):
    print(f"\nID: {doc_id}")
    print(f"Text preview: {doc[:150]}...")