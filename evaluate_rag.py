import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

load_dotenv()

# --- 1. SET YOUR MANUAL HYPERPARAMETERS HERE ---
MANUAL_CHUNK_SIZE = 1000
MANUAL_CHUNK_OVERLAP = 200
MANUAL_TOP_K = 4
# -----------------------------------------------

print(f"\n--- Testing RAG with Chunk: {MANUAL_CHUNK_SIZE}, Overlap: {MANUAL_CHUNK_OVERLAP}, Top-K: {MANUAL_TOP_K} ---")

# 2. Rebuild the Vector Store with current parameters
docs = load_all_documents("data")
embedder = EmbeddingPipeline(chunk_size=MANUAL_CHUNK_SIZE, chunk_overlap=MANUAL_CHUNK_OVERLAP)
vector_store = FaissVectorStore(persist_dir="faiss_store_eval")

chunks = embedder.chunk_documents(docs)
chunk_texts = [doc.page_content for doc in chunks]
embeddings = embedder.embed_text(chunk_texts)
vector_store.build_from_documents(embeddings, chunks)

rag = RAGSearch(vector_store=vector_store, embedding_pipeline=embedder)

# 3. Create your "Golden Dataset"
# Add 5-10 highly specific questions based on your 6 ML/AI PDFs
eval_data = {
    "question": [
        "What is scaled dot-product attention?",
        "What problem do residual connections solve in Deep ResNets?",
        "How does XGBoost handle missing values?"
    ],
    "ground_truth": [
        "Scaled dot-product attention is an attention mechanism where the dot products of queries and keys are divided by the square root of their dimension (d_k) to prevent vanishing gradients, followed by a softmax function.",
        "Residual connections solve the degradation problem in deep neural networks, where adding more layers causes training accuracy to saturate and then degrade rapidly.",
        "XGBoost handles missing values by assigning them to a default direction in each tree node, learning the best direction from the data during training."
    ]
}

print("\nRunning RAG queries to collect answers and contexts...")
answers = []
contexts = []

# 4. Generate Answers and Retrieve Contexts
for q in eval_data["question"]:
    # Get answer
    result = rag.search_and_summarize(q, top_k=MANUAL_TOP_K)
    answers.append(result["answer"])
    
    # Ragas needs the exact text chunks retrieved to score precision/recall
    query_vector = embedder.embed_text([q])[0]
    retrieved_chunks = vector_store.similarity_search(query_vector, top_k=MANUAL_TOP_K)
    contexts.append([chunk.page_content for chunk in retrieved_chunks])

# 5. Format and Evaluate
eval_dataset = Dataset.from_dict({
    "question": eval_data["question"],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": eval_data["ground_truth"]
})

print("\nStarting Ragas Evaluation (This may take a minute)...")
result = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

# 6. Print Results
print("\n=========================================")
print("Evaluation Results:")
print(f"Faithfulness (No Hallucinations): {result['faithfulness']:.4f}")
print(f"Answer Relevance (On Topic):      {result['answer_relevancy']:.4f}")
print(f"Context Precision (Right Chunks): {result['context_precision']:.4f}")
print(f"Context Recall (Complete Info):   {result['context_recall']:.4f}")
print("=========================================")
print(f"Overall Average Score: {(result['faithfulness'] + result['answer_relevancy'] + result['context_precision'] + result['context_recall']) / 4:.4f}")
print("=========================================\n")