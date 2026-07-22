import streamlit as st
import ollama
import fitz
import chromadb

# ---- Setup ----

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="papers")

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def is_citation_heavy(chunk):
    bracket_count = chunk.count('[') + chunk.count(']')
    digit_count = sum(c.isdigit() for c in chunk)
    return (bracket_count > 15) or (digit_count / max(len(chunk), 1) > 0.15)

# ---- Interface ----

st.title("NanoMind")
st.write("Hello! This is my research assistant app.")

st.sidebar.header("Navigation")

filter_citations = st.sidebar.checkbox("Filter out reference-list chunks", value=True)

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            paper_text = ""
            for page in doc:
                paper_text += page.get_text()

            chunks = chunk_text(paper_text)

            skipped = 0
            with st.spinner(f"Embedding {uploaded_file.name}..."):
                for i, chunk in enumerate(chunks):
                    if filter_citations and is_citation_heavy(chunk):
                        skipped += 1
                        continue
                    embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
                    collection.add(
                        ids=[f"{uploaded_file.name}_{i}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"source": uploaded_file.name}]
                    )

            st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"{uploaded_file.name} loaded! {len(chunks)} chunks found "
                       f"({skipped} citation-heavy chunks skipped), and stored.")
        else:
            st.info(f"{uploaded_file.name} is already processed and stored.")

# ---- Search scope ----

all_metadata = collection.get()['metadatas']
paper_names = sorted(set(m['source'] for m in all_metadata)) if all_metadata else []

scope = st.selectbox("Search scope:", ["All papers"] + paper_names)

question = st.text_input("Ask a question about the paper:")

if st.button("Submit"):
    if question:
        query_filter = None if scope == "All papers" else {"source": scope}

        with st.spinner("Searching document..."):
            question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3,
                where=query_filter
            )
            relevant_chunks = results['documents'][0]
            relevant_sources = results['metadatas'][0]

            labeled_chunks = []
            for chunk, meta in zip(relevant_chunks, relevant_sources):
                labeled_chunks.append(f"[Source: {meta['source']}]\n{chunk}")

            context = "\n\n".join(labeled_chunks)

        with st.spinner("Thinking..."):
            prompt = f"""Here are relevant excerpts from one or more documents, each labeled with its source:

{context}

Based ONLY on these excerpts, answer this question: {question}

When you use information from an excerpt, cite the source filename in parentheses right after the relevant sentence, like this: (Source: filename.pdf)

If the excerpts do not contain enough information to answer the question, say so explicitly rather than using outside knowledge."""
            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'user', 'content': prompt}
            ])

        st.write(response.message.content)

        with st.expander("See the source excerpts used"):
            for chunk, meta in zip(relevant_chunks, relevant_sources):
                st.caption(f"Source: {meta['source']}")
                st.write(chunk)
                st.divider()
    else:
        st.write("Please type a question first.")