from typing import Annotated

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.db import init_db
from src.repositories.chat import ChatRepository
from src.services.chat import ChatService


def get_chat_service(
    db: Annotated[AsyncIOMotorDatabase, Depends(init_db)],
) -> ChatService:
    chat_repository = ChatRepository(db)
    chat_service = ChatService(chat_repository)
    return chat_service
