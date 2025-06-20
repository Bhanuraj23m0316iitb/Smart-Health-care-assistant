import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTORSTORE_DIR = "vector_db_dir"
PDF_DIR = "data"


if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
    print("✅ Vector store already exists. Skipping embedding...")
else:
    print("🚀 Loading PDFs and generating vector embeddings...")

    
    loader = DirectoryLoader(path=PDF_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"📄 Total pages loaded: {len(documents)}")

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    text_chunks = text_splitter.split_documents(documents)

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
   
    print("✅ Documents vectorized.")
