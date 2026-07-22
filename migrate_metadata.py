import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="papers")

all_data = collection.get()
ids = all_data['ids']
metadatas = all_data['metadatas']

updated_ids = []
updated_metadatas = []

for doc_id, metadata in zip(ids, metadatas):
    if metadata is None or 'source' not in metadata:
        # id looks like "filename.pdf_3" - strip off the trailing _<number>
        source = doc_id.rsplit('_', 1)[0]
        updated_ids.append(doc_id)
        updated_metadatas.append({'source': source})

if updated_ids:
    collection.update(
        ids=updated_ids,
        metadatas=updated_metadatas
    )
    print(f"Updated metadata for {len(updated_ids)} chunks.")
else:
    print("Nothing needed updating — all chunks already have source metadata.")