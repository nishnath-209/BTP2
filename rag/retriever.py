import faiss
import pickle
from sentence_transformers import SentenceTransformer

# DATA_FOLDER = "p/data"
# RAG_FOLDER = "p/rag"
# SCRIPTS_FOLDER = "p/scripts"
# LLM_FOLDER = "p/llm"
# PIPELINE_FOLDER = "p/pipeline"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("rag/index.faiss")

with open(( "rag/docs.pkl"), "rb") as f:
    docs = pickle.load(f)


# def retrieve(query, k=3):

#     query_embedding = model.encode([query])

#     distances, indices = index.search(query_embedding, k)

#     results = []

#     for i in indices[0]:
#         results.append(docs[i])

#     return results

def retrieve(query, k=3, debug=True):

    print("\n==============================")
    print("RAG STEP 1: USER QUERY")
    print(query)

    # encode query
    query_embedding = model.encode([query])

    if debug:
        print("\nRAG STEP 2: QUERY EMBEDDING SHAPE")
        print(query_embedding.shape)

    # search FAISS
    distances, indices = index.search(query_embedding, k)

    if debug:
        print("\nRAG STEP 3: FAISS DISTANCES")
        print(distances)

        print("\nRAG STEP 4: DOCUMENT INDICES")
        print(indices)

    results = []

    for idx in indices[0]:
        results.append(docs[idx])

    if debug:
        print("\nRAG STEP 5: RETRIEVED DOCUMENTS")
        for i, r in enumerate(results):
            print(f"\n--- Document {i+1} ---")
            print(r)

    print("==============================\n")

    return results