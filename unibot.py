import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Cache expensive resources
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",  # More powerful model
        model_kwargs={'device': 'cpu'},  # Adjust based on available hardware
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_chat_model():
    """Initialize the Ollama chat model"""
    return Ollama(model="llama2", temperature=0.3)

def process_documents(uploaded_files):
    """Process uploaded documents using appropriate loaders"""
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Select loader based on file type
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.lower().endswith(".docx"):
                loader = UnstructuredFileLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
            
            docs.extend(loader.load())
    return docs

def create_vector_store(docs, embeddings):
    """Create Chroma vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./db"
    )
    return vectorstore

# Streamlit app configuration
st.set_page_config(page_title="University RAG Chatbot")
st.title("🎓 University RAG Chatbot")

# Initialize session state
if "vectorstore_exists" not in st.session_state:
    st.session_state.vectorstore_exists = os.path.exists("./db")

# Document upload section
uploaded_files = st.file_uploader(
    "Upload university documents (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

# Handle vector store creation/rebuilding
rebuild = False
if uploaded_files:
    if st.session_state.vectorstore_exists:
        rebuild = st.checkbox("Rebuild vector store with new documents?", value=False)
        if rebuild:
            if os.path.exists("./db"):
                import shutil
                shutil.rmtree("./db")
                st.session_state.vectorstore_exists = False
                st.info("[INFO] Vector store deleted")

    if not st.session_state.vectorstore_exists or rebuild:
        st.info("[INFO] Creating new vector store from uploaded files")
        with st.spinner("Processing documents..."):
            docs = process_documents(uploaded_files)
            embeddings = load_embedding_model()
            create_vector_store(docs, embeddings)
            st.session_state.vectorstore_exists = True
            st.success("[INFO] Vector store created successfully")

# Load existing vector store if available
vectorstore = None
if st.session_state.vectorstore_exists:
    st.info("[INFO] Loading existing vector store")
    try:
        embeddings = load_embedding_model()
        vectorstore = Chroma(
            persist_directory="./db",
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")

# Initialize QA system if vector store exists
qa_chain = None
if vectorstore:
    llm = load_chat_model()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Chat interface
if qa_chain:
    query = st.text_input("Enter your question about university documents:")
    if query:
        with st.spinner("Searching for answers..."):
            result = qa_chain({"query": query})
            
        st.markdown(f"**Answer:**\n\n{result['result']}")
        
        # Display source documents
        with st.expander("View source documents"):
            for idx, doc in enumerate(result['source_documents']):
                st.subheader(f"Source {idx + 1}")
                st.write(f"**Document:** {doc.metadata.get('source', 'Unknown')}")
                st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                st.text(doc.page_content)
                st.write("---")
else:
    st.info("Upload documents and create a vector store to begin querying")