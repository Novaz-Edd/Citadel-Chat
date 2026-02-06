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
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

import bcrypt
from jose import jwt, JWTError

from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
SECRET_KEY = "citadel_master_key_change_me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

UPLOAD_DIR = "citadel_vault"       # Secure storage for PDFs
PERSIST_DIR = "citadel_memory"     # Vector DB storage
MODEL_NAME = "qwen2.5:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 3. DATABASE SETUP (SQLite)
# -------------------------------------------------------------------------
DB_URL = "sqlite:///./citadel_users.db"

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


# Create all tables in the database
Base.metadata.create_all(bind=engine)


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

    # Create and return the access token
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


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
    access_token = create_access_token(data={"sub": guest_username})
    return {"access_token": access_token, "token_type": "bearer"}


# =========================================================================
# STEP 4 - RAG & FILE UPLOAD LOGIC (The Brain)
# =========================================================================

# -------------------------------------------------------------------------
# 1. DATA VALIDATION
# -------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


# -------------------------------------------------------------------------
# 2. UPLOAD ENDPOINT (The Vault)
# -------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload a PDF to the user's private vault and index it for RAG."""
    # Save the uploaded file to the vault
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and split the PDF into chunks
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Tag every chunk with the owner's ID for data isolation
    for chunk in chunks:
        chunk.metadata["owner_id"] = current_user.id
        chunk.metadata["source"] = file.filename

    # Create embeddings and store in ChromaDB
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Explicitly persist to disk (ensures data survives restarts)
    try:
        vector_db.persist()
    except AttributeError:
        pass  # Newer Chroma versions auto-persist

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "chunks_indexed": len(chunks),
        "owner": current_user.username,
    }


# -------------------------------------------------------------------------
# 3. CHAT ENDPOINT (The Intelligence)
# -------------------------------------------------------------------------
@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """Ask a question against the user's private knowledge vault."""
    # Initialize embeddings and load the vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Create a retriever with an owner_id filter for data isolation
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"owner_id": current_user.id},
        }
    )

    # Initialize the LLM
    llm = OllamaLLM(model="qwen2.5:1.5b")

    # Define the prompt template
    prompt_template = PromptTemplate(
        template=(
            "Answer the question strictly based on the provided context. "
            "If the context does not contain the answer, say "
            "'I don't have enough information in your vault to answer that.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # Helper to format retrieved documents into a single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the LCEL RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Also retrieve source documents separately for the response
    source_docs = retriever.invoke(request.question)
    answer = rag_chain.invoke(request.question)

    # Extract unique source filenames
    sources = list(
        {doc.metadata.get("source", "unknown") for doc in source_docs}
    )

    return {
        "answer": answer,
        "sources": sources,
    }


# =========================================================================
# STEP 5 - MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
