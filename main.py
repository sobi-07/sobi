import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="Day 5 GenAI API", version="1.0.0")

class ChatRequest(BaseModel):
    question: str

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=api_key,
    )

@app.get("/")
def root():
    return {
        "message": "Day 5 GenAI API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        response = get_llm().invoke(question)
        content = response.content
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            answer = "".join(parts)
        else:
            answer = str(content)
        return {"question": question, "answer": answer.strip()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {type(exc).__name__}: {exc}",
        )
