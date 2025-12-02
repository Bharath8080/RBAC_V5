# 🧠 AGENTO: Enterprise AI Knowledge Assistant

![Agento Banner](public/main.gif)

**Agento** is a production-ready, enterprise-grade AI assistant designed to streamline internal knowledge management. Built with **Streamlit** and powered by **Google Gemini**, it allows companies to ingest PDF documents, index them using **Qdrant**, and interact with their knowledge base via text or voice.

---

## ✨ Key Features

- **🔐 Secure Authentication**: 
  - Role-based access control (Admin vs. Employee).
  - Secure company registration and team joining flows.
  - Password hashing with `bcrypt`.

- **📚 RAG (Retrieval-Augmented Generation)**:
  - **Upload & Index**: Admins can upload PDF documents (HR policies, Technical Docs, etc.).
  - **Vector Search**: Uses `Qdrant` and `HuggingFace Embeddings` for semantic search.
  - **Smart Answers**: Powered by `Gemini-2.0-Flash` for accurate, context-aware responses.

- **🗣️ AI Call Mode**:
  - **Voice-to-Voice**: Speak directly to Agento using `SpeechRecognition`.
  - **Audio Response**: Hear responses via `gTTS` (Google Text-to-Speech).
  - **Hands-Free**: Continuous conversation loop.

- **📊 Visual Thinking**:
  - Automatically generates **Mermaid.js** flowcharts for process-related questions.
  - Interactive modal popups for diagrams.

- **📈 Admin Dashboard**:
  - Track document uploads and team member stats.
  - Manage knowledge base categories.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based UI)
- **Database**: SQLite (Local relational DB)
- **Vector DB**: [Qdrant](https://qdrant.tech/) (Cloud or Local)
- **LLM**: Google Gemini 2.0 Flash (via `langchain-google-genai`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Audio**: `SpeechRecognition` (STT), `gTTS` (TTS)
- **Orchestration**: [LangChain](https://www.langchain.com/)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google Cloud API Key (for Gemini)
- A Qdrant Cloud URL & API Key

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/agento.git
cd agento
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your credentials:

```ini
# Google Gemini API
GOOGLE_API_KEY="your_google_api_key_here"

# Qdrant Vector Database
QDRANT_URL="your_qdrant_url_here"
QDRANT_API_KEY="your_qdrant_api_key_here"

# (Optional) MongoDB URI is no longer needed as we migrated to SQLite
# MONGO_URI="..." 
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📖 Usage Guide

### For Admins
1. **Register**: Go to "Register Company" tab.
2. **Dashboard**: Navigate to "Admin Dashboard".
3. **Upload**: Select a category (e.g., "HR") and upload PDF documents.
4. **Share ID**: Copy the "Workspace ID" from your profile and share it with employees.

### For Employees
1. **Join**: Go to "Join Team" tab and enter the Workspace ID provided by your admin.
2. **Workspace**: Go to "Employee Workspace" to chat with the AI.
3. **Call Mode**: Switch to "AI Call Mode" for voice interaction.

---

## 📂 Project Structure

```
Agento/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (GitIgnored)
├── agento.db              # Local SQLite database (Auto-generated)
├── public/                # Static assets (images, gifs)
├── faiss_indexes/         # (Legacy) Local vector stores
└── packages.txt           # System-level dependencies
```

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

**Built with ❤️ by the Agento Team**
