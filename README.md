<div align="center">

# 🏰 Citadel-Chat

### Your Private Knowledge Fortress

A secure, self-hosted **RAG (Retrieval-Augmented Generation)** chat application powered by local LLMs. Upload your documents, ask questions, and get intelligent answers — all without sending a single byte to external AI providers.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama)](https://ollama.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
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
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **100% Private** | All processing happens locally — your data never leaves your server |
| 📄 **PDF Upload & Indexing** | Upload PDFs and automatically chunk, embed, and index them for search |
| 🤖 **Conversational RAG** | Ask follow-up questions naturally — the system rewrites vague queries using chat history |
| 👥 **Multi-User Auth** | JWT-based authentication with role-based access (Admin, Client, Guest) |
| 🛡️ **Admin Dashboard** | Full user management: create, delete, reset passwords, change roles |
| 🎨 **Modern UI** | Dark-themed, responsive single-page frontend |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| ☁️ **Render.com Blueprint** | One-click deploy to Render with persistent storage |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (SPA)                  │
│            index.html • Dark Theme UI            │
├──────────────────┬──────────────────────────────┤
│   Auth Module    │      Chat Interface           │
│  Login/Register  │   Upload • Chat • Admin       │
└──────────┬───────┴──────────┬───────────────────┘
           │    REST API      │
┌──────────▼──────────────────▼───────────────────┐
│              FastAPI Backend                      │
├──────────────┬──────────────┬───────────────────┤
│  Auth Layer  │  RAG Pipeline│   Admin API        │
│  JWT + bcrypt│  LangChain   │   User CRUD        │
└──────┬───────┴──────┬───────┴──────┬────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐
│   SQLite    │ │  ChromaDB  │ │  Ollama  │
│  Users/Auth │ │  Vectors   │ │  LLM +   │
│             │ │            │ │ Embeddings│
└─────────────┘ └────────────┘ └──────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JS (Single-Page App) |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **LLM** | Ollama → qwen2.5:1.5b (configurable) |
| **Embeddings** | Ollama → nomic-embed-text |
| **Vector Store** | ChromaDB (persistent, file-based) |
| **RAG Framework** | LangChain (core, community, ollama, chroma) |
| **Database** | SQLAlchemy + SQLite |
| **Auth** | JWT (python-jose) + bcrypt |
| **Containerization** | Docker, Docker Compose |
| **Deployment** | Render.com (Blueprint ready) |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** — [Download](https://python.org/downloads/)
- **Ollama** — [Download](https://ollama.com/download)
- **Git** — [Download](https://git-scm.com/)

### Option 1: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Novaz-Edd/Citadel-Chat.git
cd Citadel-Chat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull required models
ollama serve                        # In a separate terminal
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# 4. Run the application
python main.py
```

Open **http://localhost:8000** in your browser.

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker compose up --build
```

The app will be available at **http://localhost:8000**. Models are downloaded automatically on first startup.

### Option 3: Deploy to Render

1. Fork this repository
2. Connect your GitHub account to [Render.com](https://render.com)
3. Create a **New Blueprint** and select the repo
4. Render auto-detects `render.yaml` and configures everything
5. Wait for the build + first model download (~5 min)

> **Note:** Requires at least the **Standard** plan (2 GB RAM) to run the LLM.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `citadel_master_key_change_me` | JWT signing secret (change in production!) |
| `PORT` | `8000` | Server port |
| `MODEL_NAME` | `qwen2.5:1.5b` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DATA_DIR` | `.` (script directory) | Persistent data directory (DB, uploads, vectors) |

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Create a new account |
| `POST` | `/token` | Login and receive JWT |
| `POST` | `/guest` | Login as guest |
| `GET`  | `/me` | Get current user profile |

### RAG / Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF document |
| `POST` | `/chat` | Send a message (with RAG context) |
| `GET`  | `/history` | Get conversation history |
| `DELETE` | `/history` | Clear conversation history |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/users` | List all users |
| `DELETE` | `/admin/users/{id}` | Delete a user |
| `PUT` | `/admin/users/{id}/reset-password` | Reset a user's password |
| `PUT` | `/admin/users/{id}/role` | Change a user's role |

> Full interactive API docs available at `/docs` (Swagger UI).

---

## 🛡️ Admin Panel

A default admin account is created on first startup:

| Field | Value |
|-------|-------|
| **Username** | `admin` |
| **Password** | `admin123` |

> ⚠️ **Change the default admin password immediately in production.**

The admin panel provides:
- 📊 **Dashboard** — User statistics (total, clients, guests, admins)
- 👤 **User Management** — View all registered users
- 🔑 **Reset Passwords** — Generate new passwords for any user
- 🏷️ **Role Management** — Promote/demote users (admin, client, guest)
- 🗑️ **Delete Users** — Remove accounts (admin accounts are protected)

---

## 📁 Project Structure

```
Citadel-Chat/
├── main.py                 # FastAPI backend (auth, RAG, admin, chat)
├── requirements.txt        # Python dependencies
├── Dockerfile              # All-in-one Docker image (Python + Ollama)
├── docker-compose.yml      # Local development with Docker
├── render.yaml             # Render.com deployment blueprint
├── start.sh                # Container startup script
├── reset_db.py             # Utility: reset database
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker ignore rules
├── frontend/
│   └── index.html          # Single-page frontend application
├── citadel_vault/          # Uploaded PDF documents (gitignored)
├── citadel_memory/         # ChromaDB vector store (gitignored)
└── citadel_users.db        # SQLite database (gitignored)
```

---

## 🖼️ Screenshots

<div align="center">

| Login Screen | Chat Interface |
|:---:|:---:|
| Dark themed auth with Sign In, Register, Guest, and Admin login | Conversational RAG with PDF upload support |

| Admin Panel | PDF Upload |
|:---:|:---:|
| User management dashboard with stats and controls | Drag-and-drop PDF upload with processing status |

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Novaz-Edd](https://github.com/Novaz-Edd)**

*Your data. Your models. Your fortress.*

</div>