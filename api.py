"""Optional REST API:  uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector import check_answer

app = FastAPI(title="Stale-Answer Detector")


class CheckRequest(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/check")
def check(req: CheckRequest):
    try:
        return check_answer(req.question, req.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
