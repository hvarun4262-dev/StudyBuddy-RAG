import glob
from langchain_community.document_loaders import PyPDFLoader

def load_all_documents(data_dir: str):
    """Loads all AI/ML PDF papers from the specified directory using LangChain."""
    documents = []
    pdf_files = glob.glob(f"{data_dir}/*.pdf")
    
    print(f"Found {len(pdf_files)} PDF files in '{data_dir}'. Loading...")
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            # Tag metadata so we know which paper the chunk came from
            doc.metadata["source_file"] = pdf_path.split("/")[-1]
        documents.extend(docs)
        
    print(f"Successfully loaded {len(documents)} total pages.")
    return documents