from pathlib import Path
from typing import Annotated, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.dependencies import get_chat_service
from src.models.core import PyObjectId
from src.services.chat import ChatService, final

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/")
async def get_chats(chat_service: Annotated[ChatService, Depends(get_chat_service)]):
    return await chat_service.get_chats()


@router.post("/")
async def create_chat(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    name: Optional[str] = Form(default=None),
    urls: str = Form(...),
    file: UploadFile = File(...),
):
    # write file to disk
    if file.filename:
        filename = Path(file.filename)
        filename.touch(exist_ok=True)
        async with aiofiles.open(f"./uploads/{file.filename}", "wb+") as out_file:
            content = await file.read()
            await out_file.write(content)

    return await chat_service.create_chat(
        name=name, urls=urls.split(","), file=f"./uploads/{file.filename}"
    )


@router.get("/query/{chat_id}")
async def query(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    chat_id: str,
    query: str,
):
    try:
        chat = await chat_service.get_chat_summary(chat_id)
        chat_summary = "\n".join(url.description for url in chat.urls)  # type: ignore

        outputs = final(chat_id, question=query, summary=chat_summary)
        await chat_service.add_message(PyObjectId(chat_id), outputs)
        return outputs

    except Exception as exc:
        print(exc)
