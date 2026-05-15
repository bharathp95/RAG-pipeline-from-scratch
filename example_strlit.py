
import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# setup
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="mps",
    local_files_only=True
)
client = chromadb.PersistentClient(path="pdf_db")
groq_client = Groq(api_key="")


def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def store_chunks(chunks):
    client.delete_collection("pdf_chunks")
    collection = client.get_or_create_collection("pdf_chunks")
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    return collection


def retrieve(query, collection, top_k=3):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]


def ask(query, collection):
    chunks = retrieve(query, collection)
    context = "\n\n".join(chunks)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using only the context provided."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return response.choices[0].message.content


# UI
st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        collection = store_chunks(chunks)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

    query = st.text_input("Ask a question about your PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = ask(query, collection)
        st.write(answer)





# streamlit run app.py --logger.level=error