import os
import json
import argparse
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import chromadb

def process_qa_json(file_path, filename):
    """Process Q&A JSON files into document chunks"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        documents = []
        for qa in qa_pairs:
            content = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            metadata = {
                "source": filename,
                "question": qa['question'],
                "answer": qa['answer'],
                "content_type": "qa_pair"
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    except Exception as e:
        print(f"Error processing Q&A JSON {filename}: {str(e)}")
        return []

def process_structured_text(content, filename):
    """Process TXT files with custom delimiters"""
    separators = [
        "\n## ",  # Main headings
        "\n** ",  # Subheadings
        "\n----",  # Section breaks
        "\n\n",
        "\n",
        ". "
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=separators,
        keep_separator=True
    )
    
    docs = [Document(page_content=content, metadata={"source": filename})]
    return text_splitter.split_documents(docs)

def process_documents():
    """Process documents with enhanced format support"""
    docs = []
    input_dir = "data"
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Data folder '{input_dir}' not found")

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loaded = loader.load()
            elif filename.lower().endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                loaded = process_structured_text(content, filename)
            elif filename.lower().endswith(".docx"):
                loader = UnstructuredFileLoader(file_path)
                loaded = loader.load()
            elif filename.lower().endswith(".json"):
                if "qa" in filename.lower():
                    loaded = process_qa_json(file_path, filename)
                else:
                    # Fallback for other JSON formats
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                    loaded = [Document(
                        page_content=str(content),
                        metadata={"source": filename}
                    )]
            else:
                continue
                
            for doc in loaded:
                if 'source' not in doc.metadata:
                    doc.metadata["source"] = filename
            docs.extend(loaded)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    return docs

def create_vector_store(docs, output_dir):
    """Create Chroma vector store with optimized splitting"""
    if not docs:
        raise ValueError("No documents processed")
    
    # Unified text splitter for all documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=[
            "\n## ", "\n** ", "\n----",  # TXT structure
            "\nQuestion: ", "\nAnswer: ",  # Q&A structure
            "\n\n", "\n", ". "  # General separators
        ],
        keep_separator=True
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