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


