import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Changed import
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import chromadb
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CUSTOM_PROMPT = PromptTemplate(
    template="""<|system|>
    You are a university administrative assistant. Use the context to answer thoroughly.
    Follow these guidelines:
    1. For exact question matches, use the provided answer verbatim
    2. Combine information from multiple sources when needed
    3. Structure complex answers with bullet points
    4. Cite document sources when possible
    
    Context: {context}
    
    <|user|>
    Question: {question}
    
    <|assistant|>""",
    input_variables=["context", "question"]
)

def load_qa_chain(db_path="./db"):
    """Initialize QA system with Groq API"""
    try:
        client = chromadb.PersistentClient(path=db_path)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma(
            client=client,
            collection_name="university_docs",
            embedding_function=embeddings,
            persist_directory=db_path
        )
       
        return RetrievalQA.from_chain_type(
            llm=ChatGroq(
                temperature=0.3,
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq model
                groq_api_key=os.getenv("GROQ_API_KEY")
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "score_threshold": 0.65
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.stop()

st.set_page_config(page_title="GBPUAT Chatbot", layout="wide")
st.title(" UniBot - GBPUAT Chatbot")
st.markdown(
    """
    This chatbot can assist you with queries related to admissions, programs, and university policies of GBPUAT, Pantnagar.
    """
)

# Initialize QA system
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_qa_chain()

# Chat interface
query = st.text_input("Ask about admissions, programs, or university policies:")
if query:
    with st.spinner("Analyzing academic documents..."):
        try:
            result = st.session_state.qa_chain({"query": query})
            
            st.markdown(f"### Answer\n{result['result']}")
            
            with st.expander("📚 Source Documents"):
                for idx, doc in enumerate(result['source_documents']):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**Document {idx+1}**")
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    with col2:
                        st.text(doc.page_content[:600].strip() + "...")
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")