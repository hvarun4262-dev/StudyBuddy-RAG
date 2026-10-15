import os
import pickle
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store"):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []
        os.makedirs(self.persist_dir, exist_ok=True)

    def build_from_documents(self, embeddings, chunks):
        """Creates a FAISS IndexFlatL2 vector index and saves it locally."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = chunks
        self.save()
        print(f"Built FAISS vector index with {self.index.ntotal} vectors.")

    def save(self):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pickle"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """Loads the saved FAISS index from disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pickle")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index files not found. Run /build-index first.")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def similarity_search(self, query_embedding, top_k=4):
        """Retrieves top-k matching document chunks."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), top_k
        )
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results