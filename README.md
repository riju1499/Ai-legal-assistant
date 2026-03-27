# ⚖️ AI Legal Assistant (Nepali Legal Case Intelligence System)

An intelligent legal assistant system designed to help users understand and navigate **Nepali legal cases** through **semantic search, AI-driven chatbots, and legal strategy recommendations**.

---

# 🎯 Project Overview

AI Legal Assistant combines advanced **Artificial Intelligence techniques** with a comprehensive legal knowledge base to provide:

- 🔍 **Intelligent Case Search** – Semantic search across Nepali court cases
- 🤖 **Legal Chatbot** – Interactive AI assistant for legal inquiries
- 📑 **Case Analysis** – Detailed case information with contextual insights
- 🧠 **Strategy Generation** – AI-powered legal strategy recommendations
- 📚 **Legal Glossary** – Definitions of legal terms and concepts

---

# 🏗️ System Architecture

## Backend (`/backend`)

**Framework:** FastAPI (Python)

### AI Components

- Large Language Model (Generative AI)
- Retrieval-Augmented Generation (RAG)
- Agentic Reasoning System (LangGraph)
- Semantic Search using **Qdrant Vector Database**
- Intelligent Query Router
- Legal Strategy Generator

### Key Backend Modules

```
agent/
│
├── llm_client.py
├── rag_pipeline.py
├── agent_graph.py
├── qdrant_kb.py
├── intelligent_router.py
├── response_synthesizer.py
└── strategy_generator.py
```

| Module | Description |
|------|-------------|
| llm_client.py | Interface with the language model |
| rag_pipeline.py | Retrieves relevant legal documents |
| agent_graph.py | Multi-agent orchestration workflow |
| qdrant_kb.py | Vector database integration |
| intelligent_router.py | Query classification |
| response_synthesizer.py | Response generation |
| strategy_generator.py | Legal strategy suggestions |

---

# 💻 Frontend (`/frontend`)

**Framework:** React 18 + Vite

### Features

- Case search interface
- AI chatbot interaction
- Case detail visualization
- Legal strategy recommendations
- Responsive modern UI

### Key Pages

| Page | Description |
|-----|-------------|
| SearchPage.jsx | Case search interface |
| ChatPage.jsx | AI legal chatbot |
| CaseDetail.jsx | Detailed case view |
| StrategyPage.jsx | Legal strategy suggestions |

---

# 📚 Knowledge Base

### `/global_knowledge_base`

Contains structured legal resources including:

- Case metadata
- Court information
- Case type classification
- Verdict summaries

### `/glossary`

Contains:

- Legal terms
- Definitions
- Legal concepts

---

# 🧠 Vector Storage

### `/qdrant_storage`

Stores:

- Semantic embeddings of legal cases
- Vector-based similarity search

---

# 🔎 Search Index

### `/search_index`

Contains:

- FAISS search index
- Case corpus
- Metadata for full-text search

---

# 🚀 Getting Started

## Prerequisites

- Python **3.8+**
- Node.js **16+**
- pip or conda

---

# ⚙️ Backend Setup

### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file:

```
LLM_API_KEY=your_api_key_here
```

### Run backend server

```bash
python main.py
```

Backend will run at:

```
http://localhost:8000
```

API Documentation:

```
http://localhost:8000/docs
```

---

# 🎨 Frontend Setup

### Install dependencies

```bash
cd frontend
npm install
```

### Run development server

```bash
npm run dev
```

Frontend will run at:

```
http://localhost:5173
```

### Build for production

```bash
npm run build
```

---

# 📊 Data Processing Pipeline

The system includes a **data processing pipeline** located in:

```
/scripts
```

This pipeline handles:

- Legal case preprocessing
- Legal glossary creation
- Semantic embedding generation
- Vector database initialization
- Search index construction
- System performance monitoring

### Key Scripts

| Script | Purpose |
|------|--------|
| build_index.py | Constructs search indices |
| 06_build_semantic_index.py | Creates vector embeddings |
| 09_build_qdrant_knowledge.py | Initializes semantic database |

---

# 🔍 Query Types

The AI system intelligently handles multiple query categories:

| Query Type | Description |
|------------|------------|
| Law Explanation | Understanding laws and legal concepts |
| Case Recommendation | Finding similar cases and precedents |
| Legal Advice | Guidance for specific legal situations |
| Loophole Analysis | Identifying patterns and gaps in legal provisions |
| General Inquiry | Questions about the legal system |

---

# 🛠️ API Endpoints

| Method | Endpoint | Description |
|------|----------|-------------|
| GET | `/` | Health check |
| POST | `/search` | Semantic case search |
| POST | `/chat` | AI legal chatbot |
| POST | `/strategy` | Legal strategy generation |
| GET | `/case/{case_id}` | Case details |
| GET | `/docs` | Interactive API documentation |

---

# 📈 Metrics & Monitoring

The system includes monitoring features such as:

- Response quality metrics
- Query classification accuracy
- System health monitoring
- Performance analytics

Metrics are exported to:

```
/scripts/csv_exports/
```

---

# 💡 Technologies Used

| Category | Technology |
|--------|-------------|
| LLM | Google Generative AI |
| Vector Database | Qdrant |
| Search Engine | FAISS |
| NLP | Sentence Transformers, HuggingFace |
| Backend | FastAPI |
| Frontend | React |
| AI Orchestration | LangGraph |
| Build Tool | Vite |

---

# 📂 Project Structure

```
AI-Legal-Assistant
│
├── backend/
│   ├── agent/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── global_knowledge_base/
├── glossary/
├── qdrant_storage/
├── search_index/
└── scripts/
```

---

# 🔐 Security & Privacy

- API keys are securely managed using **environment variables**
- No personal user data is stored
- Legal case data comes from **public legal records**

---

# ⚠️ Disclaimer

This system is designed **for educational and informational purposes only**.

It should **not be considered official legal advice**.  
Users should consult **qualified legal professionals** for actual legal guidance.

---

# 👨‍💻 Author

**Riju Phaiju**

BE Computer Engineering Student  
Aspiring Software Engineer & QA Engineer

---

⭐ If you found this project useful, consider **starring the repository**.
