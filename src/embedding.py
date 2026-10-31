from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        
    def chunk_documents(self, documents):
        """Splits document pages into overlapping chunks using LangChain."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks

    def embed_text(self, text_list):
        return self.model.encode(text_list, show_progress_bar=True)