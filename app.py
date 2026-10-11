import os
import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Lab: Ask Your PDF", page_icon="📚", layout="wide")

st.title("📚 RAG Lab: Ask Your PDF")
st.caption("A beginner-friendly Retrieval-Augmented Generation (RAG) demo for AI students.")

with st.expander("What happens in a RAG pipeline?"):
    st.markdown(
        "1. **Load** text from the PDF.  2. **Chunk** the text.  "
        "3. **Index** chunks with TF-IDF.  4. **Retrieve** relevant chunks for a question.  "
        "5. **Generate** an answer grounded in the retrieved context."
    )

st.sidebar.header("Answering model")
provider = st.sidebar.radio("Choose provider", ["Groq API (cloud)", "Ollama (local)"])
groq_key = st.sidebar.text_input(
    "Groq API key (free-tier account)", type="password",
    help="Used only for Groq API requests. You can also set GROQ_API_KEY in Streamlit secrets."
)
groq_model = st.sidebar.text_input("Groq model ID", value="openai/gpt-oss-120b")
ollama_model = st.sidebar.text_input("Ollama model name", value="llama3.2")
st.sidebar.info(
    "No paid API is required for this lab. Groq's free tier has rate/token limits; "
    "Ollama runs locally but needs you to install and download a model."
)

@st.cache_data(show_spinner=False)
def extract_pdf(pdf_bytes: bytes):
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages

def make_chunks(pages, chunk_words=180, overlap_words=35):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_words]).strip()
            if piece:
                chunks.append({"page": page_num, "text": piece})
    return chunks

def retrieve(question, chunks, vectorizer, matrix, top_k=4):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in indices if scores[i] > 0]

def ask_groq(question, context, api_key, model):
    prompt = (
        "You are a helpful teaching assistant. Answer the question using only the supplied "
        "PDF context. If the context does not contain the answer, say that the PDF does not "
        "provide enough information. Cite page numbers when available.\n\n"
        f"PDF CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Answer accurately and stay grounded in the provided context."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Groq API error ({response.status_code}): {response.text[:700]}")
    return response.json()["choices"][0]["message"]["content"]

def ask_ollama(question, context, model):
    prompt = (
        "Answer using only the PDF context. If the answer is absent, say so. "
        "Cite page numbers when possible.\n\n"
        f"PDF CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama error: {response.text[:700]}")
    return response.json()["message"]["content"]

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded:
    try:
        pages = extract_pdf(uploaded.getvalue())
        if not pages:
            st.error("No selectable text was found. Scanned PDFs need OCR before this demo can read them.")
            st.stop()
        chunks = make_chunks(pages)
        if not chunks:
            st.error("Could not create text chunks from this PDF.")
            st.stop()

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([c["text"] for c in chunks])
        st.success(f"PDF loaded and indexed: {len(pages)} text pages, {len(chunks)} chunks.")
        st.caption("For this classroom demo, indexing uses TF-IDF rather than a neural embedding model.")

        question = st.text_input("Ask a question about your PDF", placeholder="What are the main ideas?")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=min(8, len(chunks)), value=min(4, len(chunks)))

        if st.button("Get answer", type="primary", disabled=not question.strip()):
            matches = retrieve(question, chunks, vectorizer, matrix, top_k)
            if not matches:
                st.warning("No relevant text was found. Try rephrasing your question.")
            else:
                context = "\n\n".join(
                    f"[Page {chunk['page']} | relevance score {score:.3f}]\n{chunk['text']}"
                    for chunk, score in matches
                )
                try:
                    with st.spinner("Generating answer..."):
                        if provider == "Groq API (cloud)":
                            key = groq_key.strip() or os.getenv("GROQ_API_KEY", "")
                            try:
                                if not key:
                                    key = st.secrets.get("GROQ_API_KEY", "")
                            except Exception:
                                pass
                            if not key:
                                st.error("Enter a Groq API key in the sidebar, or add GROQ_API_KEY to Streamlit secrets.")
                                st.stop()
                            answer = ask_groq(question, context, key, groq_model.strip())
                        else:
                            answer = ask_ollama(question, context, ollama_model.strip())
                    st.subheader("Answer")
                    st.write(answer)
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the selected model service. For Ollama, make sure Ollama is running locally.")
                except requests.exceptions.Timeout:
                    st.error("The model request timed out. Try a smaller question or a different model.")
                except Exception as exc:
                    st.error(str(exc))

        with st.expander("Inspect retrieved chunks (useful for teaching RAG)"):
            if question.strip():
                matches = retrieve(question, chunks, vectorizer, matrix, top_k)
                if matches:
                    for idx, (chunk, score) in enumerate(matches, start=1):
                        st.markdown(f"**Chunk {idx} — page {chunk['page']} — score {score:.3f}**")
                        st.write(chunk["text"])
                        st.divider()
            else:
                st.write("Enter a question to inspect the retrieved context.")
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
else:
    st.info("Upload a PDF to begin. Try a short lecture note or textbook chapter.")

st.markdown("---")
st.caption("Educational prototype. Verify answers against the source PDF; retrieval and generation can make mistakes.")
