from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime

# Import our modules
from db import init_database, log_conversation, get_all_logs, add_training_data, get_training_data
from chains import get_chatbot, reset_chatbot

# Initialize FastAPI app
app = FastAPI(
    title="Smart FAQ Chatbot API",
    description="A REST API backend using FastAPI, LangChain, and Google Gemini for FAQ chatbot functionality",
    version="1.0.0"
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

class TrainingRequest(BaseModel):
    question: str
    answer: str

class TrainingResponse(BaseModel):
    id: int
    message: str
    question: str
    answer: str

class LogResponse(BaseModel):
    id: int
    question: str
    answer: str
    timestamp: str
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the database when the app starts."""
    init_database()
    print("✅ Database initialized successfully")
    print("🤖 FAQ Chatbot API is ready!")

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the API is running."""
    return HealthResponse(
        status="healthy",
        message="Smart FAQ Chatbot API is running successfully!",
        timestamp=datetime.now().isoformat()
    )

# Main chat endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Accept a user question and return an AI-generated answer.
    Also logs the interaction to the database.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get chatbot instance and ask question
        chatbot = get_chatbot()
        answer = chatbot.ask(request.question)
        
        # Log the conversation to database
        log_conversation(request.question, answer)
        
        return QuestionResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get all logs endpoint
@app.get("/logs", response_model=List[LogResponse])
async def get_logs():
    """
    Retrieve all saved Q&A logs from the database.
    """
    try:
        logs = get_all_logs()
        return [
            LogResponse(
                id=log["id"],
                question=log["question"],
                answer=log["answer"],
                timestamp=log["timestamp"],
                session_id=log["session_id"]
            )
            for log in logs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

# Training endpoint
@app.post("/train", response_model=TrainingResponse)
async def add_training(request: TrainingRequest):
    """
    Accept manual Q&A pairs and store them as training data.
    This helps improve the chatbot's responses for specific domains.
    """
    try:
        if not request.question.strip() or not request.answer.strip():
            raise HTTPException(status_code=400, detail="Both question and answer are required")
        
        # Add training data to database
        training_id = add_training_data(request.question, request.answer)
        
        # Reset chatbot to reload training data
        reset_chatbot()
        
        return TrainingResponse(
            id=training_id,
            message="Training data added successfully",
            question=request.question,
            answer=request.answer
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add training data: {str(e)}")

# Get training data endpoint (bonus)
@app.get("/training", response_model=List[Dict])
async def get_training():
    """
    Retrieve all training data from the database.
    """
    try:
        return get_training_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve training data: {str(e)}")

# Reset conversation memory endpoint (bonus)
@app.post("/reset")
async def reset_conversation():
    """
    Reset the conversation memory of the chatbot.
    """
    try:
        reset_chatbot()
        return {"message": "Conversation memory reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {str(e)}")

# Get chatbot status endpoint (bonus)
@app.get("/status")
async def get_status():
    """
    Get the current status and memory information of the chatbot.
    """
    try:
        chatbot = get_chatbot()
        memory_info = chatbot.get_memory_summary()
        
        return {
            "status": "active",
            "memory_info": memory_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
    