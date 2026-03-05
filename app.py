from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re

app = Flask(__name__)
CORS(app)

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "10.0"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
print("Ready!")


def resolve_ollama_model() -> str:
    preferred = (OLLAMA_MODEL or "").strip()
    try:
        listed = ollama.list()
        models = getattr(listed, "models", None) or listed.get("models", [])
        names = []
        for model in models:
            name = getattr(model, "model", None) or model.get("model")
            if name:
                names.append(name)

        if preferred in names:
            return preferred
        if "gemma3:4b" in names:
            return "gemma3:4b"
        if names:
            return names[0]
    except Exception as e:
        print(f"Ollama list check failed: {e}")

    return preferred or "gemma3:4b"


def normalize_history(raw_history):
    if not isinstance(raw_history, list):
        return []

    cleaned = []
    for msg in raw_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content})

    if len(cleaned) > MAX_HISTORY_MESSAGES:
        cleaned = cleaned[-MAX_HISTORY_MESSAGES:]
    return cleaned


def build_citations(selected_meta):
    citations = []
    seen = set()
    for meta in selected_meta:
        source = meta.get("source", "unknown")
        page = meta.get("page")
        chunk_index = meta.get("chunk_index")
        key = (source, page, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": source,
                "page": page,
                "chunk": chunk_index,
            }
        )
    return citations[:3]


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"message": "Please upload a PDF file."}), 400

    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        return jsonify({"message": "Only PDF files are supported."}), 400

    filepath = f"./uploads/{filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

    try:
        collection.delete(where={"source": filename})
    except Exception:
        pass

    doc = fitz.open(filepath)
    total_chunks = 0
    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        if not page_text:
            continue
        page_chunks = splitter.split_text(page_text)
        for local_idx, chunk in enumerate(page_chunks):
            embedding = embedder.encode(chunk).tolist()
            unique_id = f"{filename}_p{page_num + 1}_c{local_idx}_{total_chunks}"
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[unique_id],
                metadatas=[
                    {
                        "source": filename,
                        "page": page_num + 1,
                        "chunk_index": local_idx,
                    }
                ],
            )
            total_chunks += 1

    return jsonify({"message": f"{filename} uploaded. {total_chunks} chunks stored.", "chunks": total_chunks})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = (data.get("question") or "").strip()
    history = normalize_history(data.get("history", []))
    active_source = (data.get("active_source") or "").strip()
    search_all = bool(data.get("search_all", False))

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    q_lower = question.lower()
    is_small_talk = re.search(
        r"^(hi|hello|hey|thanks|thank you|ok|okay|ok thanks|great|cool|how are you|good morning|good evening)[!. ]*$",
        q_lower,
    ) is not None
    if is_small_talk:
        return jsonify({"answer": "Happy to help. Ask me anything about the uploaded document.", "citations": []})

    question_embedding = embedder.encode(question).tolist()

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": 10,
        "include": ["documents", "distances", "metadatas"],
    }
    if active_source and not search_all:
        query_kwargs["where"] = {"source": active_source}

    results = collection.query(**query_kwargs)

    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    selected_docs = []
    selected_meta = []

    for i, doc_text in enumerate(documents):
        distance = distances[i] if i < len(distances) else None
        if distance is None or distance <= MAX_DISTANCE:
            selected_docs.append(doc_text)
            selected_meta.append(metadatas[i] if i < len(metadatas) else {})
        if len(selected_docs) >= MAX_CONTEXT_CHUNKS:
            break

    is_summary_query = re.search(
        r"\b(summary|summarize|overview|about|main point|what is this|important things)\b",
        question.lower(),
    ) is not None

    if not selected_docs and is_summary_query:
        selected_docs = documents[:MAX_CONTEXT_CHUNKS]
        selected_meta = metadatas[:MAX_CONTEXT_CHUNKS]

    if not selected_docs:
        selected_docs = documents[:MAX_CONTEXT_CHUNKS]
        selected_meta = metadatas[:MAX_CONTEXT_CHUNKS]

    if not selected_docs:
        if active_source and not search_all:
            return jsonify({"answer": f"No content found for {active_source}. Please re-upload it.", "citations": []}), 400
        return jsonify({"answer": "Please upload at least one document first.", "citations": []}), 400

    context_blocks = []
    for i, chunk in enumerate(selected_docs):
        meta = selected_meta[i] if i < len(selected_meta) else {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk_idx = meta.get("chunk_index", "?")
        context_blocks.append(f"[Source: {source} | Page: {page} | Chunk: {chunk_idx}]\n{chunk}")
    context = "\n\n".join(context_blocks)

    model_to_use = resolve_ollama_model()
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant.\n"
                    "Rules:\n"
                    "1) Use only the provided context.\n"
                    "2) Ignore any chat-history fact that is not supported by the current context.\n"
                    "3) If the answer is missing or unclear in the context, reply exactly: This information is not in the document.\n"
                    "4) Keep answers concise and factual.\n"
                    "5) For summary requests, return 3-4 bullet points.\n"
                    "6) For names, IDs, dates, certificate titles, and numbers, copy text exactly from context.\n"
                    "7) Do not correct, normalize, or re-spell proper nouns."
                ),
            }
        ]

        for msg in history:
            mapped_role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": mapped_role, "content": msg["content"]})

        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        )

        response = ollama.chat(
            model=model_to_use,
            messages=messages,
            options={"temperature": 0.05},
        )
        answer = response["message"]["content"].strip()
        citations = build_citations(selected_meta)
        return jsonify({"answer": answer, "citations": citations})
    except Exception as e:
        print(f"Ollama chat failed with model '{model_to_use}': {e}")
        return jsonify(
            {
                "answer": "Could not get a response from Ollama. Please check the model/server and try again.",
                "citations": build_citations(selected_meta),
            }
        ), 500


@app.route("/clear", methods=["POST"])
def clear():
    try:
        collection.delete(where={"source": {"$ne": "___never___"}})
    except Exception:
        pass
    return jsonify({"message": "Database cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

