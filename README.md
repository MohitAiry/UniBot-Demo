🧠 University Chatbot using Local LLM and Vector DB
This project is a chatbot system that runs locally using lightweight open-source LLMs and a custom vector database built from university documents. It supports Groq, OpenAI, and Hugging Face APIs for various inference and embedding tasks.

🔧 Setup Instructions
1. Create a .env File
To begin, create a .env file in the root directory with the following API keys:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
🖥️ Local Installation
Step 1: Install Ollama
Ollama is required to run inference models locally.

Step 2: Download a Local Model
You can use any LLM compatible with Ollama. For this chatbot, we recommend:

arduino
Copy
Edit
ollama run phi3:mini-128k
This model is lightweight and optimized for fast inference in chatbot scenarios.

🗃️ Creating the Vector Database
Place your documents (PDFs, text files, etc.) into the /data directory.

Run the following script to generate your vector database:

bash
Copy
Edit
python create_db.py
This will generate a /db folder that stores the vector embeddings for the chatbot to search.

💬 Running the Chatbot
(Optional) Change the LLM model name in the code where it's defined, based on your downloaded Ollama model.

Start the chatbot application:

bash
Copy
Edit
streamlit run chat.py
Open the Streamlit interface in your browser (usually at http://localhost:8501) and start chatting!

🧹 Improvements (WIP)
🔍 Switch to a more efficient embedding model for faster and more accurate vector search.

🧾 Clean and structure the unorganized university documents for better information retrieval.

📁 Project Structure
bash
Copy
Edit
.
├── data/              # Unstructured university documents
├── db/                # Generated vector database
├── create_db.py       # Script to create the vector DB
├── chat.py            # Streamlit chatbot app
├── .env               # API keys (Groq, OpenAI, Hugging Face)
└── README.md          # This file
