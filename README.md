<div align="center">

# 🏰 Citadel-Chat

### Your Private Knowledge Fortress

A secure, production-ready **RAG (Retrieval-Augmented Generation)** chat application. Upload your documents, ask questions in natural language, and receive intelligent, context-aware answers — powered by Groq's ultra-fast LLM API and local HuggingFace embeddings.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?logo=groq&logoColor=white)](https://console.groq.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Framework-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#option-1-local-development)
  - [Docker](#option-2-docker)
  - [Deploy to Render](#option-3-deploy-to-render)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [Admin Panel](#-admin-panel)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Groq-Powered LLM** | Ultra-fast inference using `llama-3.3-70b-versatile` via the Groq API |
| 📄 **PDF Upload & Indexing** | Upload PDFs and automatically chunk, embed, and index them for semantic search |
| 🧠 **Conversational Memory** | Maintains chat history and rewrites vague follow-up queries for better retrieval |
| 🔒 **Per-User Data Isolation** | Every user's documents and context are strictly separated — no cross-contamination |
| 👥 **Multi-User Auth** | JWT-based authentication with role-based access control (Admin, Client, Guest) |
| 🛡️ **Admin Dashboard** | Full user management: create, delete, reset passwords, change roles |
| 👤 **Guest Access** | Allow anonymous users to try the app without registering |
| 🎨 **Modern Dark UI** | Responsive single-page frontend with no external framework dependencies |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| ☁️ **Render.com Blueprint** | One-click deploy to Render with persistent disk storage |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                  Frontend (SPA)                   │
│          index.html  •  Vanilla JS + CSS          │
├───────────────────┬──────────────────────────────┤
│    Auth Module    │       Chat Interface          │
│  Login / Register │  Upload • Chat • Admin Panel  │
└─────────┬─────────┴──────────────┬───────────────┘
          │       REST API (JSON)   │
┌─────────▼────────────────────────▼───────────────┐
│                 FastAPI Backend                    │
├───────────────┬──────────────────┬───────────────┤
│   Auth Layer  │   RAG Pipeline   │   Admin API   │
│  JWT + bcrypt │   LangChain      │   User CRUD   │
└───────┬───────┴──────┬───────────┴───────────────┘
        │              │
┌───────▼──────┐ ┌─────▼──────────────────────────┐
│    SQLite    │ │       Vector + LLM Layer         │
│  Users / DB  │ ├───────────────┬─────────────────┤
│  Messages    │ │   ChromaDB    │   Groq API       │
└──────────────┘ │  (Embeddings) │  llama-3.3-70b  │
                 ├───────────────┴─────────────────┤
                 │  HuggingFace Embeddings          │
                 │  all-MiniLM-L6-v2 (local)        │
                 └─────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (Single-Page App) |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **LLM** | [Groq API](https://console.groq.com) — `llama-3.3-70b-versatile` (configurable) |
| **Embeddings** | HuggingFace — `all-MiniLM-L6-v2` (runs locally, no API key needed) |
| **Vector Store** | ChromaDB (persistent, file-based) |
| **RAG Framework** | LangChain (core, community, groq, huggingface, chroma) |
| **Database** | SQLAlchemy + SQLite |
| **Authentication** | JWT (`python-jose`) + bcrypt |
| **Containerisation** | Docker, Docker Compose |
| **Deployment** | Render.com (Blueprint ready) |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** — [Download](https://python.org/downloads/)
- **Groq API Key** — Free at [console.groq.com](https://console.groq.com)
- **Git** — [Download](https://git-scm.com/)

### Option 1: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Novaz-Edd/Citadel-Chat.git
cd Citadel-Chat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Open .env and add your GROQ_API_KEY

# 4. Run the application
python main.py
```

Open **http://localhost:8000** in your browser.

> On Windows, you can also double-click **`start.bat`** which validates your `.env` before launching.

### Option 2: Docker

```bash
# Set your Groq API key, then build and run
GROQ_API_KEY=your_key_here docker compose up --build
```

The app will be available at **http://localhost:8000**.

### Option 3: Deploy to Render

1. Fork this repository
2. Connect your GitHub account to [Render.com](https://render.com)
3. Create a **New Blueprint** and select this repo
4. Add `GROQ_API_KEY` as an environment variable in the Render dashboard
5. Render auto-detects `render.yaml` and configures persistent storage automatically

---

## ⚙️ Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `GROQ_API_KEY` | — | **Yes** | Your Groq API key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | No | Groq chat model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | No | HuggingFace embedding model (runs locally) |
| `SECRET_KEY` | `citadel_master_key_change_me` | **Change in prod** | JWT signing secret |
| `PORT` | `8000` | No | Server port |
| `DATA_DIR` | `.` (script directory) | No | Persistent data directory for DB, uploads, and vectors |

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/register` | None | Create a new user account |
| `POST` | `/token` | None | Login and receive a JWT access token |
| `POST` | `/guest-login` | None | Create a temporary guest session |
| `GET`  | `/me` | Bearer | Get the current user's profile |

### RAG / Chat

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/upload` | Bearer | Upload and index a PDF document |
| `POST` | `/chat` | Bearer | Send a message and receive a RAG-powered response |

### Admin (Role: admin only)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/users` | List all registered users |
| `DELETE` | `/admin/users/{id}` | Delete a user by ID |
| `PUT` | `/admin/users/{id}/reset-password` | Reset a user's password |
| `PUT` | `/admin/users/{id}/role` | Change a user's role |

> Interactive API documentation (Swagger UI) is available at **`/docs`** when the server is running.

---

## 🛡️ Admin Panel

A default admin account is created automatically on first startup:

| Field | Value |
|-------|-------|
| **Username** | `admin` |
| **Password** | `admin123` |

> ⚠️ **Change the default admin password immediately before deploying to production.**

Capabilities:
- 📊 **User Statistics** — Total users, clients, guests, and admins at a glance
- 👤 **User Management** — View all registered accounts
- 🔑 **Password Reset** — Reset any user's password
- 🏷️ **Role Management** — Promote or demote users between `admin`, `client`, and `guest`
- 🗑️ **Delete Accounts** — Remove any user (admin accounts are protected from deletion)

---

## 📁 Project Structure

```
Citadel-Chat/
├── main.py                 # FastAPI backend — auth, RAG pipeline, admin, chat
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Local Docker Compose setup
├── render.yaml             # Render.com deployment blueprint
├── start.sh                # Linux/macOS startup script
├── start.bat               # Windows startup script
├── reset_db.py             # Utility script to reset the database
├── .env                    # Local environment variables (gitignored)
├── .gitignore              # Git ignore rules
├── frontend/
│   └── index.html          # Single-page frontend application
├── citadel_vault/          # Uploaded PDF documents (gitignored)
├── citadel_memory/         # ChromaDB vector store (gitignored)
└── citadel_users.db        # SQLite user database (gitignored)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes with a clear message: `git commit -m 'feat: add my feature'`
4. **Push** to your branch: `git push origin feature/my-feature`
5. **Open** a Pull Request with a description of your changes

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Novaz-Edd](https://github.com/Novaz-Edd)**

*Your documents. Your knowledge. Your fortress.*

</div>