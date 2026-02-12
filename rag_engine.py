import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DOCS_PATH = "rag/docs"
INDEX_PATH = "rag/index"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    texts = []

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                texts.append(page.extract_text())

        elif file.endswith((".txt", ".yaml", ".yml")):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts

def build_index():
    docs = load_documents()
    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss.write_index(index, f"{INDEX_PATH}/docs.index")

    with open(f"{INDEX_PATH}/docs.pkl", "wb") as f:
        pickle.dump(docs, f)

def load_index():
    index = faiss.read_index(f"{INDEX_PATH}/docs.index")
    with open(f"{INDEX_PATH}/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

def query_docs(query, top_k=3):
    index, docs = load_index()
    q_emb = model.encode([query])
    _, I = index.search(q_emb, top_k)

    return [docs[i] for i in I[0]]

