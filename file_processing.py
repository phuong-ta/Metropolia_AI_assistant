from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models_setting import GoogleAI_llm, GoogleAI_embedding, HuggingFace_embedding_model, OpenAI_embedding
from langchain_chroma import Chroma
from typing import List
import os


# load pdf files
def load_documents(file_path: str, file_name: str) -> List:
    """Load PDF documents from a directory."""
    try:
        loader = DirectoryLoader(file_path, glob=file_name, loader_cls=PyPDFLoader)
        return loader.load()
    except Exception as e:
        print(f"Error loading files: {e}")
        return []

# split text
def split_texts(documents: List, chunk_size: int = 1500, chunk_overlap: int = 150) -> List:
    """Split documents into chunks with the given chunk size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

#add metadata
def add_metadata(chunks: List, file_name: str, description: str) -> List:
    """Add metadata (file name and description) to each chunk."""
    for chunk in chunks:
        chunk.metadata = {
            "name": file_name,
            "description": description
        }
    return chunks


def save_vector_store(chunks: List, file_name: str, db_directory: str) -> Chroma:
    """Create a vector store from the chunks and save it to the specified directory."""
    persist_path = os.path.join(db_directory, file_name)
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAI_embedding(),
            persist_directory=persist_path
        )
        return db
    except Exception as e:
        print(f"Error saving vector store: {e}")
        return None


def create_db_from_files(file_path: str, file_name: str, description: str, db_directory: str) -> Chroma:
    """Main function to load files, split text, add metadata, and create a vector store."""
    
    # Remove the file extension if present (handle other extensions as well)
    file_name = os.path.splitext(file_name)[0]

    # Step 1: Load the PDF documents
    documents = load_documents(file_path, f"{file_name}.pdf")
    if not documents:
        print("No documents were loaded. Exiting.")
        return None

    # Step 2: Split the documents into text chunks
    chunks = split_texts(documents)

    # Step 3: Add metadata to each chunk
    chunks_with_metadata = add_metadata(chunks, file_name, description)

    # Step 4: Save the chunks into a vector store
    db = save_vector_store(chunks_with_metadata, file_name, db_directory)
    
    if db:
        print(f"Vector store for {file_name} created successfully.")
    else:
        print(f"Failed to create vector store for {file_name}.")
    
    return db


# read from vector data
def read_vectors_db(db_directory) -> Chroma:
    db = Chroma(
         persist_directory = db_directory, # folder, where vectordb stored.
         embedding_function = OpenAI_embedding() # embedding model
         )
    return db
    