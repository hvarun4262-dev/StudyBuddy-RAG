import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

load_dotenv()

app = FastAPI(
    title="AI Study Buddy RAG API",
    description="Microservice built with LangChain, FAISS, and Gemini 2.0 to answer ML queries."
)

embedder = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
vector_store = FaissVectorStore(persist_dir="faiss_store")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/build-index")
def build_index():
    try:
        documents = load_all_documents("data")
        if not documents:
            raise HTTPException(status_code=400, detail="No PDF files found in data/ folder.")
        
        chunks = embedder.chunk_documents(documents)
        chunk_texts = [doc.page_content for doc in chunks]
        
        embeddings = embedder.embed_text(chunk_texts)
        vector_store.build_from_documents(embeddings, chunks)
        
        return {"status": "success", "message": f"Indexed {len(chunks)} vector chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_rag(req: QueryRequest):
    try:
        if vector_store.index is None:
            vector_store.load()
            
        rag = RAGSearch(vector_store=vector_store, embedding_pipeline=embedder)
        answer = rag.search_and_summarize(req.question, top_k=req.top_k)
        
        return {"question": req.question, "answer": answer}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Index not built. Call /build-index first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))