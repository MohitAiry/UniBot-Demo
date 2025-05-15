import os
import json
import argparse
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredFileLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb

def process_documents():
    """Process documents from 'data' folder with error handling"""
    docs = []
    input_dir = "data"
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Data folder '{input_dir}' not found")

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = UnstructuredFileLoader(file_path)
            elif filename.lower().endswith(".json"):
                # JSON format: {"content": "text", "metadata": {...}}
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=".content",
                    text_content=False
                )
            else:
                continue
                
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    return docs

def create_vector_store(docs, output_dir):
    """Create Chroma vector store with validation"""
    if not docs:
        raise ValueError("No documents processed")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". "]
    )
    texts = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    client = chromadb.PersistentClient(path=output_dir)
    Chroma.from_documents(
        client=client,
        documents=texts,
        embedding=embeddings,
        collection_name="university_docs",
        persist_directory=output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./db', 
                      help='Output directory (default: ./db)')
    args = parser.parse_args()
    
    try:
        print("Processing documents...")
        docs = process_documents()
        create_vector_store(docs, args.output)
        print(f"Vector store created at {args.output}")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        exit(1)