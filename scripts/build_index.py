import faiss
import pickle
from sentence_transformers import SentenceTransformer

# DATA_FOLDER = "p/data"
# RAG_FOLDER = "p/rag"
# SCRIPTS_FOLDER = "p/scripts"
# LLM_FOLDER = "p/llm"
# PIPELINE_FOLDER = "p/pipeline"


DATA_FILE = "rag/rag_data.txt"

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

docs = open(DATA_FILE).read().split("\n\n")

embeddings = model.encode(docs)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "rag/index.faiss")

with open("rag/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Vector index built")