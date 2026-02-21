# -*- coding: utf-8 -*-
# =========================================================================
# CITADEL-CHAT: Your Private Knowledge Fortress
# =========================================================================
# A Secure RAG SaaS Backend
# =========================================================================

# -------------------------------------------------------------------------
# 1. IMPORTS
# -------------------------------------------------------------------------
import os
import shutil
import uuid
import threading
from typing import List
from datetime import datetime, timedelta

# Load .env file automatically (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse


from sqlalchemy.orm import declarative_base, sessionmaker
import bcrypt
from jose import jwt, JWTError
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey
from pydantic import BaseModel, ConfigDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
# HuggingFaceEmbeddings is imported lazily inside get_embeddings()
# so torch/sentence-transformers are never loaded when using the Inference API

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "citadel_master_key_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Use absolute paths based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use DATA_DIR env var for persistent storage (Render disk), else local
_DATA_DIR = os.getenv("DATA_DIR", _SCRIPT_DIR)
UPLOAD_DIR = os.path.join(_DATA_DIR, "citadel_vault")
PERSIST_DIR = os.path.join(_DATA_DIR, "citadel_memory")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
PORT = int(os.getenv("PORT", "8000"))
FORCE_RESET = os.getenv("FORCE_RESET_DB", "false").lower() == "true"

# Force reset embeddings if needed (handles dimension mismatches)
if FORCE_RESET:
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"[RESET] Deleted old embeddings at {PERSIST_DIR}")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ── Production RAM guard ─────────────────────────────────────────────────
# Render free tier has 512 MB RAM. Loading sentence-transformers + PyTorch
# locally consumes ~400 MB and will OOM-kill the process.
# Set HUGGINGFACEHUB_API_TOKEN to use the HF Inference API instead (free).
# Get a free token at https://huggingface.co/settings/tokens
_IS_PRODUCTION = bool(os.getenv("DATA_DIR"))
if _IS_PRODUCTION and not HUGGINGFACE_TOKEN:
    print(
        "\n"
        + "=" * 64 + "\n"
        "  [!] PRODUCTION RAM WARNING\n"
        "  HUGGINGFACEHUB_API_TOKEN is not set.\n"
        "  Local sentence-transformers loads PyTorch (~400 MB) and\n"
        "  may OOM-kill this container on Render free (512 MB).\n"
        "  Add HUGGINGFACEHUB_API_TOKEN in your Render dashboard.\n"
        "  Free token: https://huggingface.co/settings/tokens\n"
        + "=" * 64 + "\n"
    )

# -------------------------------------------------------------------------
# EMBEDDING SINGLETON — loaded once, reused across all requests
# Thread-safe via double-checked locking so concurrent background tasks
# never race to initialise the model a second time.
# On Render free (512MB RAM): set HUGGINGFACEHUB_API_TOKEN env var to use
# the HuggingFace Inference API instead of loading the model locally.
# Locally (no token): loads the model on-device as before.
# -------------------------------------------------------------------------
_EMBEDDINGS_INSTANCE = None
_EMBEDDINGS_LOCK = threading.Lock()

def get_embeddings():
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        with _EMBEDDINGS_LOCK:
            # Re-check inside the lock (double-checked locking)
            if _EMBEDDINGS_INSTANCE is None:
                if HUGGINGFACE_TOKEN:
                    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                    _EMBEDDINGS_INSTANCE = HuggingFaceInferenceAPIEmbeddings(
                        api_key=HUGGINGFACE_TOKEN,
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                    )
                    print("[Citadel] Embeddings: HuggingFace Inference API (cloud, low RAM)")
                else:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    print("[Citadel] Embeddings: local model (dev mode)")
    return _EMBEDDINGS_INSTANCE

# Lock that serialises all ChromaDB write operations so concurrent background
# ingestion tasks never corrupt the on-disk vector store.
_CHROMA_WRITE_LOCK = threading.Lock()

# -------------------------------------------------------------------------
# CHROMA SINGLETON — one persistent client reused across all requests.
# Creating a new Chroma() on every request opens a new SQLite connection,
# which causes "database is locked" errors (reported as error code 0) when
# a background ingest task is writing at the same time.
# -------------------------------------------------------------------------
_CHROMA_INSTANCE = None
_CHROMA_INSTANCE_LOCK = threading.Lock()

def get_vectorstore() -> "Chroma":
    """Return the shared ChromaDB client, initialising it on first call."""
    global _CHROMA_INSTANCE
    if _CHROMA_INSTANCE is None:
        with _CHROMA_INSTANCE_LOCK:
            if _CHROMA_INSTANCE is None:
                _CHROMA_INSTANCE = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=get_embeddings(),
                )
                print("[Citadel] ChromaDB: persistent client initialised")
    return _CHROMA_INSTANCE

# ── Upload job status tracker ─────────────────────────────────────────────
# Lightweight in-memory map: job_id → "processing" | "done" | "error: ..."
# Safe for Render free tier (single worker process).
_ingest_jobs: dict = {}
_ingest_jobs_lock = threading.Lock()

# -------------------------------------------------------------------------
# 3. DATABASE SETUP (SQLite)
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DATA_DIR, "citadel_users.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that yields a database session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------------------------------
# 4. APP INITIALIZATION
# -------------------------------------------------------------------------
app = FastAPI(title="Citadel-Chat: Your Private Knowledge Fortress")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handler (always return JSON, never plain text) ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch any unhandled exception and return a JSON error response."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# Serve the frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# =========================================================================
# STEP 2 - USER MODEL & SECURITY (The Security Gate)
# =========================================================================

# -------------------------------------------------------------------------
# 1. DATABASE MODELS
# -------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    plan_type = Column(String, default="free")
    role = Column(String, default="client")


class Message(Base):
    """Stores chat messages for conversational memory."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(String, nullable=False)
    sender = Column(String, nullable=False)  # "user" or "ai"
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create all tables in the database
Base.metadata.create_all(bind=engine)


# ── Auto-create default admin account on startup ──
def _create_default_admin():
    """Create a default admin user if none exists."""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if not admin:
            hashed = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            admin_user = User(
                username="admin",
                hashed_password=hashed,
                role="admin",
                plan_type="admin",
            )
            db.add(admin_user)
            db.commit()
            print("[Citadel] Default admin created — username: admin / password: admin123")
        else:
            print(f"[Citadel] Admin account exists: {admin.username}")
    finally:
        db.close()

_create_default_admin()


# -------------------------------------------------------------------------
# 2. PASSWORD HASHING
# -------------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its hashed version."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


# -------------------------------------------------------------------------
# 3. JWT TOKEN LOGIC
# -------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict) -> str:
    """Create a JWT access token with an expiration time."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# -------------------------------------------------------------------------
# 4. AUTH DEPENDENCY (The Gatekeeper)
# -------------------------------------------------------------------------
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db),
):
    """Decode the JWT token and return the authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


async def check_admin(current_user: User = Depends(get_current_user)):
    """Dependency that ensures the current user has admin privileges."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


# =========================================================================
# STEP 3 - AUTH API ENDPOINTS
# =========================================================================

# -------------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS (Data Validation)
# -------------------------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    """Public user representation (excludes password)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    plan_type: str


# -------------------------------------------------------------------------
# 2. REGISTER ENDPOINT
# -------------------------------------------------------------------------
@app.post("/register")
def register(user: UserCreate, db=Depends(get_db)):
    """Register a new user in the Citadel."""
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Hash the password and create the user
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": f"User '{new_user.username}' registered successfully. Welcome to the Citadel!"}


# -------------------------------------------------------------------------
# 3. LOGIN ENDPOINT
# -------------------------------------------------------------------------
@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    """Authenticate a user and return a JWT access token."""
    # Look up the user
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password",
        )

    # Create and return the access token (include role for frontend)
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}


# -------------------------------------------------------------------------
# 4. CURRENT USER INFO ENDPOINT
# -------------------------------------------------------------------------
@app.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's profile (role, plan, etc.)."""
    return current_user


# =========================================================================
# STEP 3.5 - GUEST LOGIN (Anonymous Access)
# =========================================================================

@app.post("/guest-login", response_model=Token)
def guest_login(db=Depends(get_db)):
    """Create a temporary guest account and return a JWT token.
    
    This allows anonymous usage while preserving owner_id isolation
    for the RAG pipeline. The guest gets a real DB user under the hood.
    """
    # Generate a random guest identity
    guest_id = str(uuid.uuid4())[:8]
    guest_username = f"guest_{guest_id}"
    guest_password = str(uuid.uuid4())  # random throwaway password

    # Register the guest user in the database
    hashed_password = get_password_hash(guest_password)
    new_user = User(
        username=guest_username,
        hashed_password=hashed_password,
        plan_type="guest",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Issue a JWT token for the guest
    access_token = create_access_token(data={"sub": guest_username, "role": "guest"})
    return {"access_token": access_token, "token_type": "bearer"}


# =========================================================================
# STEP 3.6 - ADMIN PANEL (Role-Based Access)
# =========================================================================

@app.get("/admin/users", response_model=List[UserResponse])
def list_all_users(
    admin: User = Depends(check_admin),
    db=Depends(get_db),
):
    """Return a list of all registered users. Admin only."""
    users = db.query(User).all()
    return users


@app.delete("/admin/users/{user_id}")
def delete_user(
    user_id: int,
    admin: User = Depends(check_admin),
    db=Depends(get_db),
):
    """Delete a user by ID. Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete an admin account")
    # Delete user's messages too
    db.query(Message).filter(Message.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"message": f"User '{user.username}' (ID: {user_id}) deleted successfully."}


class ResetPasswordRequest(BaseModel):
    new_password: str


@app.put("/admin/users/{user_id}/reset-password")
def admin_reset_password(
    user_id: int,
    body: ResetPasswordRequest,
    admin: User = Depends(check_admin),
    db=Depends(get_db),
):
    """Reset a user's password. Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"message": f"Password for '{user.username}' has been reset."}


class ChangeRoleRequest(BaseModel):
    role: str


@app.put("/admin/users/{user_id}/role")
def change_user_role(
    user_id: int,
    body: ChangeRoleRequest,
    admin: User = Depends(check_admin),
    db=Depends(get_db),
):
    """Change a user's role. Admin only."""
    if body.role not in ("client", "admin", "guest"):
        raise HTTPException(status_code=400, detail="Role must be 'client', 'admin', or 'guest'")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = body.role
    db.commit()
    return {"message": f"User '{user.username}' role changed to '{body.role}'."}


# =========================================================================
# STEP 4 - RAG & FILE UPLOAD LOGIC (The Brain)
# =========================================================================

# -------------------------------------------------------------------------
# 1. DATA VALIDATION
# -------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


# -------------------------------------------------------------------------
# 2. BACKGROUND INGESTION TASK
# -------------------------------------------------------------------------
def _ingest_pdf(file_path: str, filename: str, owner_id: int, job_id: str) -> None:
    """Load, chunk, embed, and store a PDF in ChromaDB.

    Runs in a FastAPI BackgroundTask so the HTTP response is returned
    instantly and Render's 15-second request timeout is never hit.
    Job status is tracked in _ingest_jobs so the frontend can poll
    GET /upload/status/{job_id} to know when indexing is complete.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            print(f"[Citadel] Ingest skipped — empty PDF: {filename}")
            with _ingest_jobs_lock:
                _ingest_jobs[job_id] = "error: PDF is empty or unreadable"
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Tag every chunk with the owner's ID for strict data isolation
        for chunk in chunks:
            chunk.metadata["owner_id"] = owner_id
            chunk.metadata["source"] = filename

        # Serialise all ChromaDB writes; use the singleton client so reads
        # in /chat share the same SQLite connection and never see a locked DB.
        with _CHROMA_WRITE_LOCK:
            get_vectorstore().add_documents(chunks)

        with _ingest_jobs_lock:
            _ingest_jobs[job_id] = "done"

        print(
            f"[Citadel] Ingest complete — {len(chunks)} chunks "
            f"from '{filename}' (owner_id={owner_id}, job_id={job_id})"
        )
    except Exception as e:
        with _ingest_jobs_lock:
            _ingest_jobs[job_id] = f"error: {str(e)}"
        print(f"[Citadel] Ingest error for '{filename}': {e}")


# -------------------------------------------------------------------------
# 3. UPLOAD ENDPOINT (The Vault)
# -------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload a PDF to the user's private vault.

    The file is saved to disk immediately and the response is returned
    at once. Heavy embedding / ChromaDB work runs in a background task
    so Render's 15-second request timeout is never triggered.
    """
    # Validate file type before touching disk
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    # Save the raw file immediately — this is fast (I/O only)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    # Assign a job ID so the client can poll for completion
    job_id = str(uuid.uuid4())
    with _ingest_jobs_lock:
        _ingest_jobs[job_id] = "processing"

    # Schedule CPU-heavy ingestion to run after the response is sent
    background_tasks.add_task(
        _ingest_pdf, file_path, file.filename, current_user.id, job_id
    )

    return {
        "status": "processing",
        "job_id": job_id,
        "message": f"'{file.filename}' received. Indexing in background — poll /upload/status/{job_id} to check when ready.",
        "owner": current_user.username,
    }


# -------------------------------------------------------------------------
# 4. UPLOAD STATUS ENDPOINT
# -------------------------------------------------------------------------
@app.get("/upload/status/{job_id}")
async def upload_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll the indexing status of a previously uploaded PDF.

    Returns:
      - status: "processing" | "done" | "error: <reason>"
    """
    with _ingest_jobs_lock:
        job_status = _ingest_jobs.get(job_id)

    if job_status is None:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found. It may have expired (server restart clears job history).",
        )

    return {"job_id": job_id, "status": job_status}


# -------------------------------------------------------------------------
# 3. CHAT ENDPOINT (The Intelligence)
# -------------------------------------------------------------------------
@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Ask a question against the user's private knowledge vault.

    Features:
    - Conversational memory (last 6 messages injected into prompt)
    - Uses Groq API for LLM and HuggingFace for embeddings
    """
    try:
        # Validate GROQ API Key
        if not GROQ_API_KEY or GROQ_API_KEY == "" or GROQ_API_KEY == "your_groq_api_key_here":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY is not configured. Please open the .env file and add your Groq API key. Get a free key at https://console.groq.com"
            )
        
        # ── 1. Save the incoming user message ────────────────────────────
        user_msg = Message(
            user_id=current_user.id,
            content=request.question,
            sender="user",
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # ── 2. Build Chat History (last 6 messages) ─────────────────────
        recent_messages = (
            db.query(Message)
            .filter(Message.user_id == current_user.id)
            .order_by(Message.id.desc())
            .limit(6)
            .all()
        )
        recent_messages.reverse()

        chat_history = ""
        for msg in recent_messages:
            role = "User" if msg.sender == "user" else "AI"
            chat_history += f"{role}: {msg.content}\n"

        # ── 3. Rewrite vague follow-ups into standalone queries ─────────
        #    If the user says "and..", "more", "tell me anything", etc.,
        #    the retriever can't find relevant docs. We use the LLM to
        #    rewrite the question using chat history so retrieval works.
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

        rewrite_prompt = PromptTemplate(
            template=(
                "Given the following conversation history and a follow-up question, "
                "rewrite the follow-up question as a standalone search query that "
                "captures what the user is really asking about. "
                "If the question is already clear and specific, return it unchanged.\n\n"
                "Chat History:\n{chat_history}\n\n"
                "Follow-up Question: {question}\n\n"
                "Standalone search query:"
            ),
            input_variables=["chat_history", "question"],
        )

        search_query = request.question
        # Only rewrite if there's chat history (follow-up scenario)
        if chat_history.strip():
            try:
                rewritten = (rewrite_prompt | llm | StrOutputParser()).invoke({
                    "chat_history": chat_history,
                    "question": request.question,
                })
                rewritten = rewritten.strip()
                if rewritten:
                    search_query = rewritten
            except Exception as e:
                # Fall back to original question if rewrite fails
                print(f"Query rewrite failed: {str(e)}")

        # ── 4. RAG Pipeline ─────────────────────────────────────────────
        # Use the shared ChromaDB client — avoids concurrent SQLite conflicts.
        vectorstore = get_vectorstore()

        # Retriever with owner_id filter for data isolation
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"owner_id": current_user.id},
            }
        )

        # Prompt: document-first, falls back to general knowledge when no docs match
        prompt_template = PromptTemplate(
            template=(
                "You are a helpful, knowledgeable assistant.\n\n"
                "INSTRUCTIONS:\n"
                "1. If the 'Document Context' section below contains relevant information, "
                "use it as your PRIMARY source and answer based on it directly and thoroughly.\n"
                "2. If the 'Document Context' is empty or does not contain information relevant "
                "to the question, answer from your own general knowledge.\n"
                "3. If the user asks a follow-up (e.g. 'and?', 'more', 'continue', 'tell me more'), "
                "expand on your previous answer.\n\n"
                "Chat History:\n{chat_history}\n\n"
                "Document Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question", "chat_history"],
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Use the rewritten query for retrieval, but pass original question to prompt
        source_docs = retriever.invoke(search_query)
        context_text = format_docs(source_docs)

        answer = (prompt_template | llm | StrOutputParser()).invoke({
            "context": context_text,
            "question": request.question,
            "chat_history": chat_history,
        })

        sources = list(
            {doc.metadata.get("source", "unknown") for doc in source_docs}
        )

        # ── 5. Save AI response ─────────────────────────────────────────
        ai_msg = Message(
            user_id=current_user.id,
            content=answer,
            sender="ai",
        )
        db.add(ai_msg)
        db.commit()

        return {
            "answer": answer,
            "sources": sources,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback to Render logs for debugging
        import traceback
        print(f"[Citadel] Chat error [{type(e).__name__}]: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat request failed [{type(e).__name__}]: {str(e)}"
        )


# =========================================================================
# STEP 5 - MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
