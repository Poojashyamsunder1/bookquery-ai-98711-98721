"""
BookQuery AI FastAPI Backend

Features:
- PDF upload with text extraction
- Store and list uploaded books (Postgres)
- Ask questions about book content (LLM integration placeholder)
- Book/status endpoints
"""

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import fitz  # PyMuPDF
import datetime

# Database setup (read from env vars required by container orchestration)
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
if not POSTGRES_URL:
    raise RuntimeError("POSTGRES_URL or DATABASE_URL env var must be set.")

# SQLAlchemy setup
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# Models (SQLAlchemy)


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    filename = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    upload_timestamp = Column(DateTime, server_default=func.now())


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())


# Create all tables if they don't exist (production: use migrations)
Base.metadata.create_all(bind=engine)


# Pydantic Models


class BookOut(BaseModel):
    id: int = Field(..., description="Book ID")
    title: str = Field(..., description="Book title")
    filename: str = Field(..., description="Uploaded filename")
    upload_timestamp: datetime.datetime = Field(..., description="Upload timestamp")

    class Config:
        orm_mode = True


class BookListResponse(BaseModel):
    books: List[BookOut]


class UploadStatus(BaseModel):
    status: str
    book_id: Optional[int] = None
    detail: Optional[str] = None


class AskRequest(BaseModel):
    book_id: int = Field(..., description="The ID of the book to ask about", example=1)
    question: str = Field(..., description="Question for the book", example="Summarize chapter 1.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Model answer")


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None


# PDF text extraction function


def extract_pdf_text(file_path: str) -> str:
    """Extracts all text from a PDF file using PyMuPDF (fitz)."""
    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)


# (Placeholder) LLM QA function


def answer_question_prompt(book_text: str, question: str) -> str:
    """PUBLIC_INTERFACE
    Placeholder for LLM model. Returns a canned response."""
    # In production: call actual LLM service, e.g. OpenAI, HuggingFace, etc.
    return f"[LLM placeholder] I cannot answer '{question}' for this book yet."


# Dependency for DB session


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


openapi_tags = [
    {"name": "books", "description": "Upload and list books"},
    {"name": "question", "description": "Ask questions about a book"},
    {"name": "status", "description": "Service status"},
]

# FastAPI app
app = FastAPI(
    title="BookQuery AI Backend",
    version="1.0.0",
    description="Handles PDF upload, text extraction, and question answering over PDF books.",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PUBLIC_INTERFACE
@app.get("/", response_model=StatusResponse, tags=["status"], summary="Health check")
def health_check():
    """Returns health status of API."""
    return StatusResponse(status="Healthy", message="BookQuery AI backend running.")


# PUBLIC_INTERFACE
@app.post(
    "/upload",
    response_model=UploadStatus,
    status_code=201,
    tags=["books"],
    summary="Upload a new PDF book",
)
async def upload_book(
    file: UploadFile = File(..., description="PDF file"),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF file, extracts its text, and saves it as a book record.
    Returns the new book's id and upload status.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
        # Extract text
        try:
            text = extract_pdf_text(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        # Use the filename (minus extension) as title
        title = os.path.splitext(file.filename)[0]
        # Save to DB
        book = Book(
            title=title,
            filename=file.filename,
            text=text
        )
        db.add(book)
        db.commit()
        db.refresh(book)
        return UploadStatus(status="success", book_id=book.id)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to record book in database.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# PUBLIC_INTERFACE
@app.get(
    "/books",
    response_model=BookListResponse,
    tags=["books"],
    summary="List all uploaded books",
    description="Returns a list of books that have been uploaded.",
)
def list_books(db: Session = Depends(get_db)):
    """Lists all uploaded books."""
    books = db.query(Book).order_by(Book.upload_timestamp.desc()).all()
    return BookListResponse(books=books)


# PUBLIC_INTERFACE
@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["question"],
    summary="Ask a question about a book's content",
    description="Ask a question about an uploaded PDF's extracted content."
)
def ask_book_question(
    request: AskRequest,
    db: Session = Depends(get_db)
):
    """Ask a question about a book's content."""
    book = db.query(Book).filter_by(id=request.book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found.")

    # Call LLM or answer directly (placeholder)
    answer = answer_question_prompt(book.text, request.question)

    # Record question/answer in DB (for history/audit, not required for answer)
    new_q = Question(book_id=book.id, question=request.question, answer=answer)
    db.add(new_q)
    db.commit()

    return AskResponse(answer=answer)


# PUBLIC_INTERFACE
@app.get(
    "/status",
    response_model=StatusResponse,
    tags=["status"],
    summary="System status",
    description="Returns simple backend system status. Can be extended for more details.",
)
def get_status():
    """Returns system status."""
    return StatusResponse(status="ok", message="Service running. DB connected.")


# Utility endpoint: List all Q&A for a book (not required, but helpful for UI)


@app.get(
    "/books/{book_id}/qa",
    response_model=List[AskResponse],
    tags=["question"],
    include_in_schema=False,
)
def get_book_questions(book_id: int, db: Session = Depends(get_db)):
    qas = db.query(Question).filter_by(book_id=book_id).order_by(Question.timestamp.desc()).all()
    return [AskResponse(answer=q.answer) for q in qas]
