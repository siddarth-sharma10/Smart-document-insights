# 🧠 Smart Document Insights

An AI-powered document analysis tool that lets you chat with your PDFs using RAG (Retrieval-Augmented Generation).

## 🚀 Features
- Upload any PDF (legal contracts, research papers, financial reports)
- Ask questions in natural language
- Get accurate answers from the document
- 100% free and runs locally on your machine

## 🛠️ Tech Stack
- **Python + Flask** — Backend API
- **ChromaDB** — Vector database
- **Sentence Transformers** — Free local embeddings
- **Ollama + Gemma3** — Free local LLM
- **HTML/CSS/JS** — Frontend UI

## ⚙️ How to Run
1. Install Ollama from ollama.com
2. Run `ollama pull gemma3:4b`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open `index.html` in browser

## 📁 Project Structure
```
├── app.py          # Flask backend + RAG pipeline
├── index.html      # Chat UI frontend
├── launcher.bat    # One-click launcher (Windows)
└── 01_ingestion.ipynb  # RAG notebook
```

## 🧠 How It Works
1. PDF is uploaded and text is extracted
2. Text is split into chunks and embedded
3. Embeddings stored in ChromaDB
4. User question is embedded and matched to relevant chunks
5. Matched chunks + question sent to local LLM
6. Answer returned to user
