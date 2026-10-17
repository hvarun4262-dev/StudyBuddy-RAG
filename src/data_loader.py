import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_all_documents(data_dir: str):
    documents = []
    pdf_files = glob.glob(f"{data_dir}/*.pdf")
    
    print(f"Found {len(pdf_files)} PDF files in '{data_dir}'. Loading...")
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for document in docs:
            # Safely store the clean filename for source citation
            document.metadata["source_file"] = Path(pdf_path).name
        documents.extend(docs)
        
    print(f"Successfully loaded {len(documents)} total pages.")
    return documents