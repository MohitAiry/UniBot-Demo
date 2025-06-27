# 🎓 UniBot: University Query Chatbot for GBPUAT

UniBot is a Retrieval-Augmented Generation (RAG) chatbot built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [ChromaDB](https://www.trychroma.com/). It helps students and visitors query information about the **College of Technology, GBPUAT**, including topics like admissions, academic calendar, campus facilities, and more.

> ⚡ Ask it anything — from "What is the criteria of admission?" to "What are the courses offered in College of Technology?" — UniBot has the answer.

---

## 📌 Features

- 🔍 **Question Answering** from university documents (PDFs, DOCX, TXT, JSON Q&A)
- 💬 **Live Chat Interface** using Streamlit with conversational history
- 🧠 **Document Embedding** via HuggingFace BGE (`bge-base-en-v1.5`)
- 🧾 **Source Citation**: Each answer links back to relevant documents
- ☁️ Supports **Groq (LLM via Llama 4)** or **Ollama (commented for local use)**
- 🎨 **Dark-themed UI** with rich formatting and document insights

---

## 🗂 Project Structure
UniBot/
├── chat.py # Streamlit chat UI
├── create_db.py # Script to process documents & build ChromaDB
├── data/ # Folder containing university files (PDF, TXT, DOCX, JSON)
├── db/ # Output vector store (created after running create_db.py)
├── .env # Contains API keys (e.g., GROQ_API_KEY)
└── README.me



---

## ⚙️ Installation & Setup

### 1. Clone the repo

'''bash
git clone https://github.com/yourusername/UniBot.git

cd UniBot

### Install dependencies
Make sure you are in a virtual environment.

pip install -r requirements.txt


## Add documents
Place university documents in the ./data/ folder. Supported formats:

.pdf
.docx
.txt (structured)
.json (Q&A format)

## Create .env file if using API otherwise download any Ollama Model suited for your PC
Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_groq_key_here

## Generate the Vector Store
python create_db.py --output ./db
This builds a Chroma vector store from the documents in data/.

## Run the Chatbot
streamlit run chat.py
Open the local URL shown (usually http://localhost:8501) and start chatting!


# 🧠 How UniBot Works Behind the Interface

UniBot is an AI-powered assistant designed to help users ask natural language questions about the **College of Technology, GBPUAT**. Beneath the simple chat interface lies a robust **Retrieval-Augmented Generation (RAG)** system combining vector search, document parsing, and large language models (LLMs).

---

## 🔁 End-to-End Data Flow

### Step 1: Document Collection
All university-related documents are placed in the `data/` folder. Supported formats include:

- `.pdf` - e.g., Admission brochures, Academic calendars
- `.docx` - e.g., College rulebooks or notices
- `.txt` - Structured or free-form text files
- `.json` - Specifically formatted Q&A pairs

---

### Step 2: Document Processing (`create_db.py`)

This script performs the following:

1. **File Loading**  
   - Uses LangChain loaders like `PyPDFLoader`, `UnstructuredFileLoader`, and custom JSON parsing to extract readable text.

2. **Text Chunking**  
   - Breaks large content into manageable pieces (chunks) using `RecursiveCharacterTextSplitter`.  
   - Chunk size = 1200 characters, with overlap of 300 for better context retention.

3. **Embedding Generation**  
   - Each chunk is converted into a high-dimensional vector using the HuggingFace model:  
     `BAAI/bge-base-en-v1.5`  
   - These embeddings numerically represent the meaning of the text.

4. **Vector Storage (ChromaDB)**  
   - All vectorized chunks are saved in a **persistent Chroma database**.
   - This database allows **fast semantic search** later.

📦 Output: A folder `./db/` containing the vector index.

---

## 💬 User Interaction Flow (`chat.py`)

When a user types a question into the Streamlit UI:

### 1. **Semantic Search**
   - The query is embedded using the same HuggingFace embedding model.
   - ChromaDB retrieves the top relevant document chunks based on **vector similarity** (using MMR search and threshold filtering).

### 2. **Context Construction**
   - The retrieved chunks are compiled into a **context block**.
   - This block represents everything the LLM "knows" while answering.

### 3. **Prompt Formulation**
   - A custom LangChain `PromptTemplate` is used to instruct the LLM:
     - Be accurate and concise
     - Use document context
     - Cite sources if possible
     - Use bullet points for complex answers

### 4. **LLM Response Generation**
   - The prompt + context is sent to a **Groq-hosted LLM**:  
     `meta-llama/llama-4-scout-17b-16e-instruct` or your in system LLM.
   - The LLM formulates a fluent, structured answer based on the input.

### 5. **Response Display**
   - The answer is shown in a **styled chat bubble**.
   - Supporting documents are expandable and viewable as excerpts.
   - Full chat history is retained until manually cleared.

## 🧠 What's Really Happening
    A[User Query] --> B[Convert to Embedding]
    B --> C[Search Vector DB (Chroma)]
    C --> D[Retrieve Top Matching Chunks]
    D --> E[Compose Prompt with Context]
    E --> F[Send to LLM (Groq/LLaMA-4)]
    F --> G[Generate Final Answer]
    G --> H[Display in Streamlit UI]



