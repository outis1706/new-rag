# RAG Lab: Ask Your PDF

A small Retrieval-Augmented Generation (RAG) teaching project. Students upload a PDF, the app extracts text, splits it into overlapping chunks, retrieves relevant chunks with TF-IDF, and asks a language model to answer using that context.

## Included files

- `app.py` — Streamlit front end and complete RAG pipeline.
- `requirements.txt` — Python dependencies.
- `.gitignore` — avoids committing secrets and local environments.

## Free model options

**Groq API (best for Streamlit Community Cloud):** create an account and API key at https://console.groq.com/ and use the free tier if available to your account. The free tier has rate and token limits; it is not unlimited. Never commit an API key to GitHub.

**Ollama (local):** install Ollama from https://ollama.com/, then download a model, for example:

```bash
ollama pull llama3.2
ollama serve
```

Choose Ollama in the app's sidebar. This uses your own computer's resources and avoids per-token API charges, but it is not unlimited in speed or capacity and depends on your hardware. Ollama running on your laptop will not be reachable from a public Streamlit Cloud app.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Deploy free on GitHub + Streamlit Community Cloud

1. Create a GitHub account and create a **public** repository, e.g. `rag-pdf-lab`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to the repository root. Do not upload PDFs containing private/student data.
3. For hosted answers, create a Groq API key at https://console.groq.com/ (check the provider's current free-tier eligibility and limits).
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Click **Create app**, select your repository, branch (`main`), and entrypoint file (`app.py`), then deploy.
6. In the Streamlit app dashboard, open **Settings / Secrets** (or Advanced settings during deployment) and add:

```toml
GROQ_API_KEY = "paste-your-key-here"
```

7. Save/reboot the app, then upload a PDF and ask a question. The key is stored as a Streamlit secret, not in the GitHub repository.
8. If you do not want to use a hosted API, run the app locally with Ollama instead. A hosted Streamlit app cannot use `localhost` to reach Ollama on your personal computer.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
Dependency guidance: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Classroom exercise

Ask students to:
1. Upload a PDF and ask a factual question.
2. Inspect the retrieved chunks and page numbers.
3. Change the number of retrieved chunks (`top_k`) and compare the context.
4. Ask a question not answered in the PDF and observe whether the model admits the evidence is missing.
5. Discuss limitations: TF-IDF is a simple lexical retriever, scanned PDFs need OCR, and model answers can still be incorrect.

## Limitations

- This is a deliberately simple teaching implementation, not a production system.
- PDF text extraction works best on text-based PDFs. Scanned pages need OCR.
- TF-IDF is a baseline retrieval method, not semantic neural embeddings.
- Free hosted model services have usage limits and can change their available model IDs. If the default Groq model ID is unavailable, choose a currently supported chat model from the provider dashboard and enter its ID in the sidebar.
