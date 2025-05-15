import os
import time
import shutil
import tempfile
import atexit
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import chromadb

# Cache expensive resources
@st.cache_resource
def load_embedding_model():
    """Load the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_chat_model():
    """Initialize the chat model"""
    return Ollama(model="phi3:mini-128k", temperature=0.3)

def get_chroma_client():
    """Configure Chroma client"""
    return chromadb.PersistentClient(
        path="./db",
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

def process_documents(uploaded_files):
    """Process uploaded documents with generic handling"""
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.lower().endswith(".docx"):
                loader = UnstructuredFileLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
            
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = file.name
            docs.extend(loaded)
    return docs

def create_vector_store(docs, embeddings):
    """Create vector store with general-purpose splitting"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    )
    texts = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        client=get_chroma_client(),
        documents=texts,
        embedding=embeddings,
        collection_name="university_docs",
        persist_directory="./db"
    )
    return vectorstore

def delete_chroma_db(path="./db", max_retries=3):
    """Safely delete Chroma DB directory"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error deleting vector store: {str(e)}")
            time.sleep(1)
    return False

def expand_query(query):
    """Generic academic query expansion"""
    expansions = {
        "requirements": ["criteria", "eligibility", "prerequisites"],
        "exam": ["test", "entrance", "evaluation"],
        "admission": ["enrollment", "registration", "intake"]
    }
    for term, synonyms in expansions.items():
        if term in query.lower():
            query += " " + " ".join(synonyms)
    return query

# Streamlit app configuration
st.set_page_config(page_title="University Chatbot")
st.title("🎓 University Document Assistant")

# Initialize session state
if "vectorstore_exists" not in st.session_state:
    st.session_state.vectorstore_exists = os.path.exists("./db")

# Document upload section
uploaded_files = st.file_uploader(
    "Upload university documents (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

# Handle vector store management
rebuild = False
if uploaded_files:
    if st.session_state.vectorstore_exists:
        rebuild = st.checkbox("Rebuild vector store with new documents?", value=False)
        if rebuild and delete_chroma_db():
            st.session_state.vectorstore_exists = False
            st.rerun()

    if not st.session_state.vectorstore_exists or rebuild:
        with st.spinner("Processing documents..."):
            docs = process_documents(uploaded_files)
            embeddings = load_embedding_model()
            create_vector_store(docs, embeddings)
            st.session_state.vectorstore_exists = True
            st.success("Vector store created successfully")

# Initialize QA system
qa_chain = None
if st.session_state.vectorstore_exists:
    try:
        embeddings = load_embedding_model()
        llm = load_chat_model()
        
        vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name="university_docs",
            embedding_function=embeddings,
            persist_directory="./db"
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "score_threshold": 0.6}
        )

        prompt_template = """Analyze the context to answer the question.
        If information is unavailable, state that clearly.
        Combine relevant details from multiple sections when needed.

        Context: {context}
        Question: {question}
        Answer:"""

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")

# Chat interface
if qa_chain:
    query = st.text_input("Ask about university policies or requirements:")
    if query:
        expanded_query = expand_query(query)
        with st.spinner("Searching documents..."):
            result = qa_chain({"query": expanded_query})
        
        if result['source_documents']:
            st.markdown(f"**Answer:**\n\n{result['result']}")
            with st.expander("View source excerpts"):
                for idx, doc in enumerate(result['source_documents']):
                    st.write(f"From: {os.path.basename(doc.metadata.get('source_file', 'Unknown'))}")
                    st.text(doc.page_content[:500] + "...")
                    st.write("---")
        else:
            st.info("No relevant information found in documents")
else:
    st.info("Upload documents to begin querying")

# Cleanup on exit
atexit.register(lambda: delete_chroma_db(max_retries=2))