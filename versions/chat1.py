import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import os
from langchain.vectorstores import Chroma

def load_qa_chain(db_path):
    """Load QA system from existing vector store"""
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
        llm=Ollama(model="phi3:mini-128k", temperature=0.3),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )

st.set_page_config(page_title="GBPUAT Chatbot")
st.title("UniBot")

# Initialize system
if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = load_qa_chain("./db")
    except Exception as e:
        st.error(f"Failed to load knowledge base: {str(e)}")
        st.stop()

# Chat interface
query = st.text_input("Ask about university policies or requirements:")
if query:
    with st.spinner("Searching knowledge base..."):
        result = st.session_state.qa_chain({"query": query})
    
    st.markdown(f"**Answer:**\n\n{result['result']}")
    
    with st.expander("View source documents"):
        for idx, doc in enumerate(result['source_documents']):
            st.write(f"📄 Source {idx+1}: {doc.metadata['source']}")
            st.text(doc.page_content[:500] + "...")
            st.write("---")