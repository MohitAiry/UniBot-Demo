import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import chromadb
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

# from langchain_community.llms import Ollama

# page config
st.set_page_config(
    page_title="GBPUAT Chatbot",
    layout="wide",
    page_icon="🎓"
)

load_dotenv()

UNIVERSITY_LOGO = r"D:\UNI QUERYBOT\UniBotv2\logo.png"

# Custom CSS for dark theme
st.markdown("""
<style>
    .user-bubble {
        background-color: #2d2d2d;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 15px 15px 0 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        border: 1px solid #404040;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
        border-left: 4px solid #58a6ff;
    }
    .bot-bubble {
        background-color: #3d3d3d;
        color: #ffffff;
        padding: 1rem;
        border-radius: 15px 15px 15px 0;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #505050;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        box-shadow: -3px 3px 10px rgba(0,0,0,0.2);
        border-left: 4px solid #2ecc71;
    }
    .user-bubble span, .bot-bubble span {
        font-size: 1rem !important;
    }
    .source-box {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.95rem !important;
    }
    .st-expander > div {
        background-color: #1e1e1e !important;
        border-color: #404040 !important;
    }
    .st-expander label {
        color: #ffffff !important;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
    }
    .stMetric {
        background-color: #2d2d2d !important;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

CUSTOM_PROMPT = PromptTemplate(
    template="""<|system|>
    You are a university administrative assistant. Use the context to answer thoroughly.
    Follow these guidelines:
    1. Provide concise and accurate answers.
    2. For exact question matches, use the provided answer verbatim
    3. Combine information from multiple sources when needed
    4. Structure complex answers with bullet points
    5. Cite document sources when possible
    
    Context: {context}
    
    <|user|>
    Question: {question}
    
    <|assistant|>""",
    input_variables=["context", "question"]
)

def load_qa_chain(db_path="./db"):
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
            # llm=Ollama(model="llama3", temperature=0.3),
            # We have a choice here either using Ollama (i.e. running locally) or Groq
            llm=ChatGroq(
                temperature=0.3,
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
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

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header with logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image(UNIVERSITY_LOGO, width=100)
with col2:
    st.title("UniBot - Academic Query Assistant")

# Welcome Section
with st.container():
    st.markdown("""
    <div style="background: linear-gradient(145deg, #2d4059, #1e1e1e);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;">
        <h3 style="color: #ffffff; margin-bottom: 0.5rem;">🎓 Welcome to UniBot!</h3>
        <p style="color: #e0e0e0; font-size: 1.1rem;">
        Ask me about:<br>
        • Admissions & Programs 📚<br>
        • Campus Facilities 🏛️<br>
        • Academic Calendar 📅<br>
        • Student Services 🧑💻
        </p>
    </div>
    """, unsafe_allow_html=True)

# Initialize QA system
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_qa_chain()

# Chat container
chat_container = st.container()
sidebar = st.sidebar

# Input form with auto-submit
with st.form("chat_form", clear_on_submit=True):
    query = st.text_area(
        "Enter your question:", 
        height=100,
        key="query_input",
        help="Press Enter ⏎ to submit | Shift+Enter ⏎ for new line",
        label_visibility="collapsed",
        placeholder="Ask your queries about GBPUAT here..."
    )
    submit = st.form_submit_button("Ask Now", type="primary", use_container_width=True)

# JavaScript injection for Enter key submission
st.components.v1.html("""
<script>
const observer = new MutationObserver(() => {
    const textarea = document.querySelector('textarea[aria-label="Enter your question:"]');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if(e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const form = textarea.closest('form');
                const button = form.querySelector('button[type="primary"]');
                if(button) button.click();
            }
        });
    }
});

observer.observe(document.body, { childList: true, subtree: true });
</script>
""")

# Process query
if submit and query:
    with st.spinner("Analyzing documents..."):
        try:
            result = st.session_state.qa_chain({"query": query})
            
            # Store in chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": result['result'],
                "sources": result['source_documents']
            })
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Display chat history
with chat_container:
    st.subheader("Chat History", anchor=False)
    
    for idx, chat in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"""
        <div class="user-bubble">
            <span style="color: #58a6ff; font-size: 0.9em;">You:</span><br>
            <div style="padding: 0.5rem 0; color: #e0e0e0;">
            {chat['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot answer
        st.markdown(f"""
        <div class="bot-bubble">
            <span style="color: #2ecc71; font-size: 0.9em;">UniBot:</span><br>
            <div style="padding: 0.5rem 0; color: #ffffff;">
            {chat['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source documents
        with st.expander(f"📚 Sources for Question {idx+1}", expanded=False):
            for source_idx, doc in enumerate(chat['sources']):
                with st.container():
                    st.markdown(f"""
                    <div class="source-box">
                        <strong style="color: #58a6ff;">Source {source_idx+1}:</strong> 
                        <span style="color: #e0e0e0;">{doc.metadata.get('source', 'Unknown')}</span><br><br>
                        <em style="color: #2ecc71;">Content:</em> 
                        <div style="color: #cccccc; margin-top: 0.5rem;">
                        {doc.page_content[:500].strip() + "..."}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Enhanced Sidebar
with sidebar:
    st.header("Quick Actions")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    st.download_button(
        label="📥 Export Chat",
        data=str(st.session_state.chat_history),
        file_name="chat_history.txt"
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Quick Links")
    st.markdown("""
    - [University Website](https://www.gbpuat.ac.in)
    - [Academic Calendar](https://www.gbpuat.ac.in/calendar.html)
    - [AUAMS Portal](https://gbpuat.auams.in/)
    - [COT Time Table](https://sites.google.com/view/collegetimetabletech/home)
    """)
    
    st.markdown("---")
    
    st.info("💡 Tip: Click sources to view document excerpts")

# Footer
st.markdown("---")
st.caption("GBPUAT Academic Assistant v2.0 ")
