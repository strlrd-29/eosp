from fastapi import APIRouter, File, Form, UploadFile

from src.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/")
async def create_chat(urls: list[str] = Form(...), file: UploadFile = File(...)):
    await ChatService.embed_urls(urls)
