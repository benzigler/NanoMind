import ollama
import numpy as np

def get_embedding(text):
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return response['embedding']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentence1 = "The dog ran fast across the yard."
sentence2 = "A canine sprinted quickly through the grass."
sentence3 = "I love eating pizza on Fridays."

emb1 = get_embedding(sentence1)
emb2 = get_embedding(sentence2)
emb3 = get_embedding(sentence3)

print("Similarity between sentence 1 and 2 (should be HIGH):", cosine_similarity(emb1, emb2))
print("Similarity between sentence 1 and 3 (should be LOW):", cosine_similarity(emb1, emb3))