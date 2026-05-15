# Chat with your PDF

A RAG (Retrieval Augmented Generation) pipeline that lets you upload a PDF and ask questions about it using an LLM.

---

## What is RAG?

RAG stands for **Retrieval Augmented Generation**. Instead of sending the entire PDF to the LLM (which is expensive and hits token limits), you:

1. Split the PDF into small chunks
2. Store those chunks as embeddings in a vector database
3. When the user asks a question, retrieve only the most **relevant** chunks
4. Send those chunks + the question to the LLM

This way the LLM only sees relevant content, not the whole PDF.

---

## The Full Pipeline

```
PDF uploaded
    ↓
extract text (PyMuPDF)
    ↓
split into chunks
    ↓
embed each chunk (SentenceTransformer)
    ↓
store in ChromaDB
    ↓
user asks a question
    ↓
embed the question
    ↓
search ChromaDB → top 3 relevant chunks
    ↓
send chunks + question to Groq LLM
    ↓
answer displayed on Streamlit
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `streamlit` | Frontend UI |
| `fitz` (PyMuPDF) | Extract text from PDF |
| `chromadb` | Vector database to store and search embeddings |
| `sentence-transformers` | Convert text to embeddings |
| `groq` | Call the LLM API |

---

## Code Breakdown

### 1. Setup

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="mps",
    local_files_only=True
)
```

Loads the embedding model. Three things happening here:

- `all-MiniLM-L6-v2` — a lightweight pre-trained model that converts text into 384 numbers (embeddings)
- `device="mps"` — uses Apple Silicon GPU instead of CPU for faster encoding
- `local_files_only=True` — uses the already downloaded model, no internet needed

```python
client = chromadb.PersistentClient(path="pdf_db")
```

Creates or opens a local vector database saved in a folder called `pdf_db`. Persistent means it survives after the program closes.

```python
groq_client = Groq(api_key="your_api_key")
```

Initializes the Groq client to call the LLM later.

---

### 2. extract_text()

```python
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

- `fitz.open(stream=...)` — opens the PDF from memory (Streamlit uploads files as streams, not file paths)
- `for page in doc` — loops through every page
- `page.get_text()` — extracts raw text from that page
- Returns one big string of all the text from the entire PDF

---

### 3. split_chunks()

```python
def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

You can't embed the entire PDF text at once — it's too large. So you split it into smaller pieces called chunks.

- `chunk_size=500` — each chunk is 500 characters long
- `overlap=50` — each chunk shares 50 characters with the previous chunk

Why overlap? To avoid losing context at the edges. Without overlap a sentence could get cut in half between two chunks:

```
chunk 1: "...the function returns a"
chunk 2: "value based on the input..."   ← overlap saves this connection
```

- `start = end - overlap` — this is what creates the overlap, stepping back 50 characters before starting the next chunk

---

### 4. store_chunks()

```python
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
```

- `client.delete_collection(...)` — wipes the old PDF's chunks so they don't mix with the new PDF
- `client.get_or_create_collection(...)` — creates a fresh collection
- `enumerate(chunks)` — gives both the index `i` and the chunk text so you can create unique ids
- `model.encode(chunk).tolist()` — converts the chunk text into 384 numbers (embedding). `.tolist()` converts it from numpy array to plain Python list because ChromaDB requires that
- `collection.add(...)` — stores three things together:
  - `ids` — unique label like `chunk_0`, `chunk_1`
  - `embeddings` — the 384 numbers used for searching
  - `documents` — the original text returned after a match is found

---

### 5. retrieve()

```python
def retrieve(query, collection, top_k=3):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]
```

This is the core of RAG — finding relevant chunks for the user's question.

- `model.encode(query)` — converts the user's question into an embedding (same 384 numbers)
- `collection.query(...)` — ChromaDB compares the query embedding against every stored chunk embedding using cosine similarity and returns the closest matches
- `n_results=3` — return the top 3 most relevant chunks
- `results["documents"][0]` — the `[0]` removes the outer list ChromaDB wraps results in (it supports multiple queries at once, so results are nested)

**Why cosine similarity?** It measures the angle between two vectors. Vectors pointing in the same direction = similar meaning = small angle = score close to 1.0. Unrelated text points in different directions = large angle = score close to 0.

---

### 6. ask()

```python
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
```

- `retrieve(query, collection)` — gets the top 3 relevant chunks
- `"\n\n".join(chunks)` — joins the 3 chunks into one string with blank lines between them
- The system prompt tells the LLM to only answer from the provided context — this prevents hallucination
- The user message contains both the context (retrieved chunks) and the question
- `model="openai/gpt-oss-120b"` — OpenAI's open weight 120B parameter model running on Groq's infrastructure at 500+ tokens/second

---

### 7. Streamlit UI

```python
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
```

- `st.file_uploader(...)` — renders a file upload button, only accepts PDFs
- `if uploaded_file:` — everything inside only runs after a file is uploaded
- `st.spinner(...)` — shows a loading animation while processing
- `st.success(...)` — shows a green success message with chunk count
- `st.text_input(...)` — renders a text box for the user's question
- `if query:` — only runs after the user types something and hits enter
- `st.write(answer)` — displays the LLM's answer

---

## Key Concepts Summary

| Concept | What it means |
|---|---|
| Embedding | A list of numbers that represents the meaning of text |
| Vector Database | A database that stores and searches embeddings |
| Cosine Similarity | Measures how similar two embeddings are (0 to 1) |
| Chunking | Splitting large text into smaller pieces for embedding |
| Overlap | Shared characters between chunks to preserve context |
| RAG | Retrieve relevant chunks → pass to LLM → get accurate answer |

