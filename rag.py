from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load document
def load_doc(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 2. Chunk text
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 3. Main pipeline
def main():
    text = load_doc("data/notes.txt")
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        q_emb = model.encode([query])
        _, idx = index.search(q_emb, k=3)

        print("\nAnswer (retrieved context):")
        for i in idx[0]:
            print("-", chunks[i])

if __name__ == "__main__":
    main()
